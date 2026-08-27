"""Answers "which TF" for a region the ChromBPNet branch has already
flagged as accessible / differentially accessible between cell types.

Deliberately reuses the same JASPAR-PFM scanning already built and
validated for the has-ChIP-seq branch (motif_scoring.py) rather than
standing up TF-MoDISco / the full chrombpnet motif-hit-calling toolchain --
that's a much heavier, separate piece of infrastructure (de novo motif
discovery from DeepSHAP contribution scores) that could be added later if
this simpler direct-PWM-scan approach isn't precise enough. This module
answers "does any of our 198 known TF motifs match here, and how well" --
not "discover a novel motif we didn't already know about".

Uses a shuffled-sequence background/null model (see significance_score
below) rather than raw match score, specifically because an earlier
un-normalized version of this module was tested on real data and found
dominated by SP5/MEIS1 hits on almost every region regardless of
direction -- a strong sign those particular JASPAR PFMs are short/
low-information and match broadly by chance. Raw top-score alone can't
tell a real hit from a promiscuous one; z-score against a matched-
composition background can.
"""
import json
import zlib

import numpy as np

from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks


def _all_tf_names(jaspar_cache_path: str):
    with open(jaspar_cache_path) as f:
        return list(json.load(f).keys())


def _raw_score(seq: str, ppm) -> float:
    _, v_fwd = best_match(seq, ppm)
    _, v_rev = best_match(reverse_complement(seq), ppm)
    return max(v_fwd, v_rev)


def _mono_shuffle(seq: str, rng: np.random.RandomState) -> str:
    """Mono-nucleotide shuffle: same base composition (so GC content is
    identical to the real sequence -- controls for the confound where a
    motif just looks like "matches high-GC sequence" rather than a real
    specific pattern), randomized order (destroys any positional motif).
    Not dinucleotide-preserving (a stricter null that also controls for
    CpG-type local biases) -- see _dinuc_shuffle for that; default here
    remains mono-shuffle since nothing observed so far needed the
    stricter (and slower) alternative."""
    arr = np.array(list(seq))
    rng.shuffle(arr)
    return "".join(arr)


def _dinuc_shuffle(seq: str, rng: np.random.RandomState, max_attempts: int = 50) -> str:
    """Dinucleotide-preserving shuffle: preserves not just base composition
    but the exact frequency of every consecutive base pair (controls for
    local biases like CpG depletion that mono-shuffle doesn't). Classic
    Eulerian-path-based approach (Altschul & Erikson 1985), implemented
    here via retry-until-valid rather than the more elaborate
    Wilson-algorithm construction some reference implementations use --
    simpler to verify correct, and cheap to retry at these sequence
    lengths (168-2114bp): build the multiset of consecutive-base edges,
    shuffle each node's outgoing edge order, greedily walk from seq[0]
    consuming edges, and only accept the result if it used every edge and
    ends on seq[-1] (the only two conditions required for a valid
    Eulerian trail). Falls back to a mono-shuffle if no valid trail is
    found within max_attempts (should not happen in practice for real
    DNA sequences, which always contain repeated dinucleotides giving many
    valid trails, but a bounded retry avoids ever hanging)."""
    if len(seq) < 3:
        return seq
    edges_by_source = {}
    for i in range(len(seq) - 1):
        edges_by_source.setdefault(seq[i], []).append(seq[i + 1])

    for _ in range(max_attempts):
        shuffled = {src: rng.permutation(targets).tolist() for src, targets in edges_by_source.items()}
        cursors = {src: 0 for src in shuffled}
        path = [seq[0]]
        node = seq[0]
        ok = True
        for _ in range(len(seq) - 1):
            options = shuffled.get(node)
            cursor = cursors.get(node, 0)
            if options is None or cursor >= len(options):
                ok = False
                break
            path.append(options[cursor])
            cursors[node] = cursor + 1
            node = options[cursor]
        if ok and node == seq[-1] and len(path) == len(seq):
            return "".join(path)
    return _mono_shuffle(seq, rng)


_SHUFFLE_FNS = {"mono": _mono_shuffle, "dinuc": _dinuc_shuffle}


def significance_score(seq: str, ppm, n_shuffles: int = 100, rng=None, shuffle_method: str = "mono") -> tuple:
    """Returns (raw_score, empirical_pvalue) for how well ppm matches seq
    relative to a background of n_shuffles shufflings of the same
    sequence. shuffle_method='mono' (default) preserves base composition
    only; 'dinuc' additionally preserves dinucleotide frequencies (a
    stricter null, ~2-3x slower) -- see _dinuc_shuffle docstring for when
    that stricter control matters.

    Uses an empirical p-value (fraction of null shuffles scoring >= the
    real sequence, standard +1/+1 correction) rather than a z-score:
    z-scores blow up to meaningless, unstable values when the null
    distribution's variance is near zero -- which happens often here,
    since many JASPAR motifs are specific/long enough that almost no
    random shuffle matches them at all (raw=0.00, std=0.00), which was
    producing nonsense z-scores like 257.9 in testing. Empirical p-value
    degrades gracefully in that regime instead (bottoms out at a legitimate
    1/(n_shuffles+1), doesn't explode)."""
    if rng is None:
        rng = np.random.RandomState(0)
    shuffle_fn = _SHUFFLE_FNS[shuffle_method]
    real = _raw_score(seq, ppm)
    null_scores = np.array([_raw_score(shuffle_fn(seq, rng), ppm) for _ in range(n_shuffles)])
    pvalue = (1 + np.sum(null_scores >= real)) / (n_shuffles + 1)
    return real, pvalue


def best_matching_tfs(chrom: str, center: int, cfg, window: int = 168, top_n: int = 5,
                       n_shuffles: int = 100, min_raw_score: float = 0.05, shuffle_method: str = "mono"):
    """Scan a genomic window against all 198 cached JASPAR PFMs, return the
    top_n best-matching TFs ranked by empirical p-value against a
    shuffled-sequence background, after dropping any candidate whose raw
    match score is below min_raw_score -- a good p-value on an
    essentially-zero raw score (a motif that matches almost nothing,
    anywhere) isn't a meaningful "hit" even if technically no shuffle beat
    it either.

    This does NOT tell you whether that TF is actually bound (needs
    ChIP-seq or a trained context model for that) or expressed in this
    cell type (needs RNA-seq) -- it only tells you whether the DNA
    sequence itself looks like an unusually strong match for that TF's
    known binding motif, relative to sequence of the same base composition."""
    from pyfaidx import Fasta
    genome = Fasta(cfg.genome_fasta)
    half = window // 2
    seq = genome[chrom][center - half:center + half].seq.upper()
    if len(seq) != window:
        return []

    # deterministic seed derived from (chrom, center) -- NOT Python's
    # built-in hash(), which is salted per-process by default
    # (PYTHONHASHSEED) specifically so string hashing isn't predictable;
    # that salt makes this seed (and therefore the shuffled-background
    # p-values below) different every run unless PYTHONHASHSEED happens to
    # be fixed in the environment, silently breaking the bit-exactness
    # this pipeline documents elsewhere.
    rng = np.random.RandomState(zlib.crc32(f"{chrom}:{center}".encode()) % (2**31))
    results = []
    for tf in _all_tf_names(cfg.jaspar_pfm_cache):
        try:
            ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
        except Exception:
            continue
        if len(ppm) < 4 or len(ppm) > window:
            continue
        raw, pvalue = significance_score(seq, ppm, n_shuffles=n_shuffles, rng=rng, shuffle_method=shuffle_method)
        if raw < min_raw_score:
            continue
        results.append((tf, raw, pvalue))

    results.sort(key=lambda x: x[2])  # ascending p-value = most significant first
    return results[:top_n]


def annotate_regions(regions_df, cfg, top_n: int = 3, n_shuffles: int = 100, cell_types=None):
    """regions_df: DataFrame with 'chrom' and 'center' columns (e.g. the
    output of chrombpnet_scoring.scan_atac_peaks). Adds a 'top_tf_matches'
    column: 'TF1(raw=0.82,p=0.01,HepG2_tpm=5.4,K562_tpm=97.2),...' ranked
    by empirical p-value against a composition-matched shuffled background.
    If cell_types is given (e.g. ['HepG2','K562']), each candidate TF is
    also annotated with its RNA-seq TPM in each cell type (via
    expression.py) -- this is the piece that turns "motif matches here"
    into "...and the TF is actually expressed", closing the gap the
    original raw-score-only version couldn't address at all. A TF with a
    great p-value but ~0 TPM in the relevant cell type is a motif match
    that almost certainly isn't biologically doing anything there.

    This is the slow step (198 TFs x (1 + n_shuffles) PWM scans per
    region) -- meant for a small, already-interesting subset (e.g. the
    top/bottom N differential regions), not a full-scale scan."""
    from . import expression

    annotations = []
    for _, row in regions_df.iterrows():
        hits = best_matching_tfs(row["chrom"], int(row["center"]), cfg, window=cfg.model.seq_len,
                                  top_n=top_n, n_shuffles=n_shuffles)
        parts = []
        for tf, raw, p in hits:
            s = f"{tf}(raw={raw:.2f},p={p:.3f}"
            if cell_types:
                for ct in cell_types:
                    tpm = expression.get_tpm(tf, ct, cfg)
                    s += f",{ct}_tpm={tpm:.1f}"
            s += ")"
            parts.append(s)
        annotations.append(",".join(parts))
    regions_df = regions_df.copy()
    regions_df["top_tf_matches"] = annotations
    return regions_df
