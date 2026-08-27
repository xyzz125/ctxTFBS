"""Parameterized version of the paper's data-prep chain
(0_cal_TF_max_prob -> 1_selec_negative_samples -> 2_gen_max168_gc_chrstrand_files
-> 3_cal_and_add_histome -> 4_gen_h5_chrstrand_gc_histome), for any TF in any
configured cell type, instead of the original scripts' hardcoded
TF_list=['GATA2'] + hardcoded absolute paths.

Same algorithm as the paper end to end: find each TF's best ChIP-seq-peak
motif match as the 'core', extract a cfg.model.seq_len-bp window around it
(168bp default), mask the core, GC/histone-match negatives from ATAC peaks
lacking ChIP-seq signal, and pack into an h5 file the training step
consumes.
"""
import pickle
from pathlib import Path

import h5py
import numpy as np
from numba import jit
from pyfaidx import Fasta

from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks


def _step_dir(cfg, cell_type: str) -> Path:
    return cfg.work_dir_path("data_prep", cell_type)


def step0_tf_max_prob(tf: str, cell_type: str, cfg):
    """For every ATAC peak, record the best PPM match value for this TF."""
    out = _step_dir(cfg, cell_type) / f"step0_{tf}.pkl"
    if out.exists():
        return out

    ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
    ct = cfg.cell_type(cell_type)
    genome = Fasta(cfg.genome_fasta)

    chrs, starts, ends, max_values = [], [], [], []
    with open(ct.atac_bed) as f:
        for line in f:
            parts = line.split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            seq = genome[chrom][start:end].seq.upper()
            _, v_fwd = best_match(seq, ppm)
            _, v_rev = best_match(reverse_complement(seq), ppm)
            chrs.append(chrom)
            starts.append(start)
            ends.append(end)
            max_values.append(max(v_fwd, v_rev))

    with open(out, "wb") as f:
        pickle.dump({"chrs": chrs, "starts": starts, "ends": ends, "max_values": max_values}, f)
    return out


def step1_negative_samples(tf: str, cell_type: str, cfg):
    """Rank ATAC peaks by PPM match score, excluding any peak overlapping
    a real ChIP-seq peak for this TF -- these become negative-sample
    candidates ('looks like the motif, but ChIP-seq says no binding')."""
    out = _step_dir(cfg, cell_type) / f"step1_{tf}_negative.bed"
    if out.exists():
        return out

    with open(step0_tf_max_prob(tf, cell_type, cfg), "rb") as f:
        d = pickle.load(f)
    chrs = np.array(d["chrs"])
    starts = np.array(d["starts"])
    ends = np.array(d["ends"])
    values = np.array(d["max_values"])

    ct = cfg.cell_type(cell_type)
    chip_bed = Path(ct.chip_bed_dir) / f"{cell_type}_ChIP_{tf}.bed"
    if not chip_bed.exists():
        raise FileNotFoundError(
            f"No bundled ChIP-seq bed for TF='{tf}' in cell_type='{cell_type}' at {chip_bed}. "
            f"This TF+cell-type combination isn't available for the has-ChIP-seq branch."
        )

    unselectable = np.zeros(len(chrs), dtype=bool)
    with open(chip_bed) as f:
        for line in f:
            parts = line.split()
            c, s, e = parts[0], int(parts[1]), int(parts[2])
            unselectable |= (chrs == c) & (s <= ends) & (e >= starts)

    score = values - unselectable.astype(float)
    order = np.argsort(-score)
    with open(out, "w") as f_out:
        for i in order:
            if score[i] > 0:
                f_out.write(f"{chrs[i]}\t{starts[i]}\t{ends[i]}\t{score[i]}\n")
    return out


@jit(nopython=True)
def _seq_to_onehot(seq_list, length):
    data = np.zeros((len(seq_list), length, 4))
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i in range(len(seq_list)):
        for j in range(length):
            ch = seq_list[i][j]
            if ch == "N":
                continue
            data[i][j][idx[ch]] = 1
    return data


def _safe_pos_orig(parts, seq) -> int:
    """Best-effort 'expected motif position' hint, used only to break ties
    when multiple positions score equally well -- never required for
    correctness. Bed files aren't consistent about where (or whether) this
    hint lives: the paper's own bundled files put a numeric offset in
    column 4, while standard ENCODE narrowPeak downloads put the peak
    *name* there (usually '.') and the actual summit offset in column 10
    instead. Try both conventions; if neither parses, fall back to the
    sequence center rather than crashing -- a tie-break default of 'assume
    the middle' is always reasonable, silently wrong ChIP-seq bed
    conventions are not."""
    for idx in (3, 9):
        if len(parts) > idx:
            try:
                return int(parts[idx])
            except ValueError:
                continue
    return len(seq) // 2


def _windowed_bed(genome, ppm, motif_len, bed_lines_iter, pos_orig_fn, out_path, seq_len: int):
    """Shared logic for extracting a seq_len-bp window (cfg.model.seq_len,
    168bp default) centered on the best PPM match within each input
    region, masking the core motif, and writing a bed-with-sequence file.
    pos_orig_fn(fields) -> the 'expected' motif position within the region
    (used only to break position ties)."""
    with open(out_path, "w") as f_out:
        for line in bed_lines_iter:
            parts = line.split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            seq = genome[chrom][start:end].seq.upper()
            pos_orig = pos_orig_fn(parts, seq)

            pos_fwd, v_fwd = best_match(seq, ppm, pos_orig)
            seq_rev = reverse_complement(seq)
            pos_rev, v_rev = best_match(seq_rev, ppm, len(seq_rev) - pos_orig)

            if v_fwd >= v_rev:
                center = start + pos_fwd
                window = genome[chrom][center - seq_len // 2:center + seq_len // 2].seq.upper()
                w_start, w_end, strand = center - seq_len // 2, center + seq_len // 2, "+"
            else:
                center = end - pos_rev
                window_fwd = genome[chrom][center - seq_len // 2:center + seq_len // 2].seq.upper()
                window = reverse_complement(window_fwd)
                w_start, w_end, strand = center - seq_len // 2, center + seq_len // 2, "-"

            if len(window) != seq_len:
                continue
            half = seq_len // 2
            m = motif_len
            core_start = half - m // 2
            masked = window[:core_start] + "N" * m + window[core_start + m:]
            gc = sum(c in "GC" for c in masked) / sum(c in "ACGT" for c in masked)
            f_out.write(f"{chrom}\t{w_start}\t{w_end}\t{strand}\t{gc}\t{masked}\t{window[core_start:core_start+m]}\n")


def step2_windows(tf: str, cell_type: str, cfg):
    pos_out = _step_dir(cfg, cell_type) / f"step2_{tf}_pos.bed"
    neg_out = _step_dir(cfg, cell_type) / f"step2_{tf}_neg.bed"
    if pos_out.exists() and neg_out.exists():
        return pos_out, neg_out

    ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
    genome = Fasta(cfg.genome_fasta)
    ct = cfg.cell_type(cell_type)
    chip_bed = Path(ct.chip_bed_dir) / f"{cell_type}_ChIP_{tf}.bed"
    seq_len = cfg.model.seq_len

    with open(chip_bed) as f:
        lines = f.readlines()
    _windowed_bed(genome, ppm, len(ppm), lines, pos_orig_fn=_safe_pos_orig, out_path=pos_out, seq_len=seq_len)

    with open(step1_negative_samples(tf, cell_type, cfg)) as f:
        lines = f.readlines()
    _windowed_bed(genome, ppm, len(ppm), lines,
                  pos_orig_fn=lambda parts, seq: len(seq) // 2,
                  out_path=neg_out, seq_len=seq_len)
    return pos_out, neg_out


def _overlaps(intervals_by_chrom, chrom, start, end) -> bool:
    for s, e in intervals_by_chrom.get(chrom, []):
        if max(start, s) < min(end, e):
            return True
    return False


def step3_add_histone(tf: str, cell_type: str, cfg):
    pos_out = _step_dir(cfg, cell_type) / f"step3_{tf}_pos.bed"
    neg_out = _step_dir(cfg, cell_type) / f"step3_{tf}_neg.bed"
    if pos_out.exists() and neg_out.exists():
        return pos_out, neg_out

    ct = cfg.cell_type(cell_type)

    def load_intervals(bed_path):
        d = {}
        with open(bed_path) as f:
            for line in f:
                c, s, e = line.split()[:3]
                d.setdefault(c, []).append((int(s), int(e)))
        return d

    me3 = load_intervals(ct.h3k4me3_bed)
    me1 = load_intervals(ct.h3k4me1_bed)

    pos_in, neg_in = step2_windows(tf, cell_type, cfg)
    for src, dst in [(pos_in, pos_out), (neg_in, neg_out)]:
        with open(src) as f_in, open(dst, "w") as f_out:
            for line in f_in:
                parts = line.strip().split("\t")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                o3 = int(_overlaps(me3, chrom, start, end))
                o1 = int(_overlaps(me1, chrom, start, end))
                f_out.write("\t".join(parts) + f"\t{o3}\t{o1}\n")
    return pos_out, neg_out


def step4_h5_dataset(tf: str, cell_type: str, cfg):
    out_h5 = _step_dir(cfg, cell_type) / f"step4_{tf}.h5"
    if out_h5.exists():
        return out_h5

    pos_bed, neg_bed = step3_add_histone(tf, cell_type, cfg)

    def read_bed(path):
        rows = [line.strip().split("\t") for line in open(path)]
        gc = [float(r[4]) for r in rows]
        seq = [r[5] for r in rows]
        hist = [r[7] + r[8] for r in rows]
        return gc, seq, hist, rows

    pos_gc, pos_seq, pos_hist, pos_rows = read_bed(pos_bed)
    neg_gc, neg_seq, neg_hist, neg_rows = read_bed(neg_bed)

    # match negatives to the positive GC/histone distribution (paper's own
    # binning approach: 50 GC bins, 4 histone-state buckets)
    n_bins = 50
    pos_bin_counts = np.zeros(n_bins, dtype=int)
    for g in pos_gc:
        pos_bin_counts[min(int(g * n_bins), n_bins - 1)] += 1
    pos_hist_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    for h in pos_hist:
        pos_hist_counts[h] += 1

    neg_bin_counts = np.zeros(n_bins, dtype=int)
    neg_hist_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    sel_neg_seq = []
    for g, s, h in zip(neg_gc, neg_seq, neg_hist):
        b = min(int(g * n_bins), n_bins - 1)
        if neg_bin_counts[b] < pos_bin_counts[b] and neg_hist_counts[h] < pos_hist_counts[h]:
            sel_neg_seq.append(s)
            neg_bin_counts[b] += 1
            neg_hist_counts[h] += 1

    # then subsample positives down to match the (now capped) negative distribution
    rng = np.random.RandomState(cfg.training.random_seed)
    order = rng.permutation(len(pos_seq))
    pos_bin_counts_new = np.zeros(n_bins, dtype=int)
    pos_hist_counts_new = {"00": 0, "01": 0, "10": 0, "11": 0}
    sel_pos_seq = []
    for i in order:
        g, s, h = pos_gc[i], pos_seq[i], pos_hist[i]
        b = min(int(g * n_bins), n_bins - 1)
        if pos_bin_counts_new[b] < neg_bin_counts[b] and pos_hist_counts_new[h] < neg_hist_counts[h]:
            sel_pos_seq.append(s)
            pos_bin_counts_new[b] += 1
            pos_hist_counts_new[h] += 1

    import numba
    seq_len = cfg.model.seq_len
    pos_onehot = _seq_to_onehot(numba.typed.List(sel_pos_seq), seq_len) if sel_pos_seq else np.zeros((0, seq_len, 4))
    neg_onehot = _seq_to_onehot(numba.typed.List(sel_neg_seq), seq_len) if sel_neg_seq else np.zeros((0, seq_len, 4))

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(f"pos_{tf}", data=pos_onehot)
        f.create_dataset(f"neg_{tf}", data=neg_onehot)

    return out_h5


def prepare(tf: str, cell_type: str, cfg):
    """Run the full data-prep chain for tf/cell_type, reusing any cached
    intermediate step already on disk. Returns the final h5 path."""
    return step4_h5_dataset(tf, cell_type, cfg)
