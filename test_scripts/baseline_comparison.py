"""Benchmark: does the ChromBPNet branch (no-ChIP-seq) actually beat the
obvious naive alternative, and does that hold up across many TFs rather
than just the two (GATA2, HNF4A) validated so far?

Baseline (method A, "motif-only"): the standard naive approach -- take
each cell type's own called ATAC peaks, keep the ones private to that
cell type (peak in HepG2, not in K562, and vice versa), and ask whether a
TF's JASPAR motif scores higher in one cell type's private peaks than the
other's. No accessibility model, no ChromBPNet, just sequence + peak
calls -- what you'd do with bedtools + a PWM scanner and nothing else.

Pipeline (method B, "ChromBPNet"): the no-ChIP-seq branch's actual
approach (screen_tfs.py / cross_validate_gata2.py) -- score every region
with a pretrained ChromBPNet accessibility model for both cell types, and
ask whether regions with a strong motif match skew toward one cell type's
predicted accessibility over the other. This is already computed for all
198 TFs in ref_result/tf_screen_specificity_hepg2_vs_k562.csv.

Ground truth: RNA-seq TPM (ENCODE). A TF is "truly" HepG2-specific if its
own gene is more highly expressed in HepG2 than K562, and vice versa --
independent of both methods above, uses neither ATAC-seq nor ChromBPNet.

For each TF where the two cell types' TPM differ clearly (avoids scoring
noise on TFs with no real expression difference to detect), check whether
each method's predicted direction (HepG2-favoring vs K562-favoring)
matches the RNA-seq direction. Report accuracy for both methods across
the whole panel.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats

from estfbu.config import load_config
from estfbu.expression import get_tpm
from estfbu.motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks

WINDOW = 168
N_SAMPLE_PER_SET = 2000
SEED = 42
TPM_LOG2FC_THRESHOLD = 1.0  # require >=2x TPM difference to call a TF "truly" specific

cfg = load_config()
rng = np.random.default_rng(SEED)


def _load_bed(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"])
    df["center"] = (df["start"] + df["end"]) // 2
    return df


def _private_regions(bed_a, bed_b, flank=500):
    """Rows of bed_a whose center doesn't fall within `flank` bp of any
    interval in bed_b, i.e. peaks private to set A."""
    mask = np.ones(len(bed_a), dtype=bool)
    b_by_chrom = {c: g[["start", "end"]].to_numpy() for c, g in bed_b.groupby("chrom")}
    for i, row in enumerate(bed_a.itertuples()):
        intervals = b_by_chrom.get(row.chrom)
        if intervals is None:
            continue
        overlap = ((intervals[:, 0] - flank) <= row.center) & (row.center <= (intervals[:, 1] + flank))
        if overlap.any():
            mask[i] = False
    return bed_a[mask].reset_index(drop=True)


def _score_regions(tf, genome, centers_by_chrom):
    ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
    half = WINDOW // 2
    scores = []
    for chrom, center in centers_by_chrom:
        try:
            seq = genome[chrom][int(center) - half:int(center) + half].seq.upper()
        except Exception:
            continue
        if len(seq) != WINDOW:
            continue
        _, v_fwd = best_match(seq, ppm)
        _, v_rev = best_match(reverse_complement(seq), ppm)
        scores.append(max(v_fwd, v_rev))
    return np.array(scores)


print("Loading ATAC peak sets...")
hepg2_bed = _load_bed(cfg.cell_types.HepG2.atac_bed)
k562_bed = _load_bed(cfg.cell_types.K562.atac_bed)
print(f"  HepG2: {len(hepg2_bed)} peaks, K562: {len(k562_bed)} peaks")

print("Computing cell-type-private peaks (no overlap with the other cell type)...")
hepg2_private = _private_regions(hepg2_bed, k562_bed)
k562_private = _private_regions(k562_bed, hepg2_bed)
print(f"  HepG2-private: {len(hepg2_private)}, K562-private: {len(k562_private)}")

hepg2_sample = hepg2_private.sample(n=min(N_SAMPLE_PER_SET, len(hepg2_private)), random_state=SEED)
k562_sample = k562_private.sample(n=min(N_SAMPLE_PER_SET, len(k562_private)), random_state=SEED)
hepg2_regions = list(zip(hepg2_sample["chrom"], hepg2_sample["center"]))
k562_regions = list(zip(k562_sample["chrom"], k562_sample["center"]))

genome = Fasta(cfg.genome_fasta)

print("\nLoading pipeline (ChromBPNet-branch) genome-wide screen results...")
pipeline_df = pd.read_csv(Path(__file__).resolve().parents[1] / "ref_result" / "tf_screen_specificity_hepg2_vs_k562.csv")
pipeline_df = pipeline_df[pipeline_df["pvalue"].notna()].set_index("tf")

tf_names = pipeline_df.index.tolist()
print(f"{len(tf_names)} TFs with valid pipeline results\n")

rows = []
t0 = time.time()
for i, tf in enumerate(tf_names):
    hepg2_tpm = get_tpm(tf, "HepG2", cfg)
    k562_tpm = get_tpm(tf, "K562", cfg)
    if np.isnan(hepg2_tpm) or np.isnan(k562_tpm):
        continue
    log2fc = np.log2((hepg2_tpm + 0.1) / (k562_tpm + 0.1))
    if abs(log2fc) < TPM_LOG2FC_THRESHOLD:
        continue  # ambiguous ground truth, skip
    true_direction = "HepG2" if log2fc > 0 else "K562"

    try:
        a_hepg2 = _score_regions(tf, genome, hepg2_regions)
        a_k562 = _score_regions(tf, genome, k562_regions)
    except ValueError:
        continue
    if len(a_hepg2) < 100 or len(a_k562) < 100:
        continue
    a_pvalue = stats.mannwhitneyu(a_hepg2, a_k562, alternative="two-sided").pvalue
    a_direction = "HepG2" if a_hepg2.mean() > a_k562.mean() else "K562"

    b_row = pipeline_df.loc[tf]
    b_direction = "HepG2" if b_row["mean_diff_strong"] > b_row["mean_diff_rest"] else "K562"

    rows.append({
        "tf": tf,
        "hepg2_tpm": hepg2_tpm, "k562_tpm": k562_tpm, "log2fc": log2fc,
        "true_direction": true_direction,
        "baseline_motif_only_direction": a_direction, "baseline_pvalue": a_pvalue,
        "pipeline_chrombpnet_direction": b_direction, "pipeline_pvalue": b_row["pvalue"],
        "baseline_correct": a_direction == true_direction,
        "pipeline_correct": b_direction == true_direction,
    })

    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(tf_names)}] {time.time()-t0:.0f}s elapsed", flush=True)

result_df = pd.DataFrame(rows)
out_path = Path(__file__).resolve().parents[1] / "ref_result" / "baseline_comparison_motif_only_vs_chrombpnet.csv"
result_df.to_csv(out_path, index=False)

n = len(result_df)
baseline_acc = result_df["baseline_correct"].mean()
pipeline_acc = result_df["pipeline_correct"].mean()
print(f"\n{n} TFs with unambiguous RNA-seq ground truth (|log2FC| >= {TPM_LOG2FC_THRESHOLD}) "
      f"and valid scores in both methods")
print(f"\nAccuracy vs RNA-seq ground truth:")
print(f"  Baseline (motif-only, private ATAC peaks):        {baseline_acc:.1%} ({result_df['baseline_correct'].sum()}/{n})")
print(f"  Pipeline (ChromBPNet-branch, genome-wide):         {pipeline_acc:.1%} ({result_df['pipeline_correct'].sum()}/{n})")

# McNemar's test on paired correct/incorrect calls
both_correct = ((result_df["baseline_correct"]) & (result_df["pipeline_correct"])).sum()
only_baseline = ((result_df["baseline_correct"]) & (~result_df["pipeline_correct"])).sum()
only_pipeline = ((~result_df["baseline_correct"]) & (result_df["pipeline_correct"])).sum()
neither = ((~result_df["baseline_correct"]) & (~result_df["pipeline_correct"])).sum()
print(f"\n2x2: both correct={both_correct}, only baseline={only_baseline}, "
      f"only pipeline={only_pipeline}, neither={neither}")
if only_baseline + only_pipeline > 0:
    mcnemar_p = stats.binomtest(only_pipeline, only_baseline + only_pipeline, 0.5).pvalue
    print(f"McNemar exact test (pipeline vs baseline, discordant pairs only): p={mcnemar_p:.3g}")

print(f"\nFull per-TF results saved to {out_path}")
