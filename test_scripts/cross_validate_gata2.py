"""Cross-validation between the two branches: does the chrombpnet branch's
raw genome-wide accessibility signal independently agree with what the
chipseq branch (real ChIP-seq-trained models) and RNA-seq expression both
already say about GATA2 -- that it's K562-specific, not HepG2-specific?

Method: for every region already scored in the full genome-wide ChromBPNet
scan, independently score how well it matches GATA2's JASPAR motif (same
validated PPM-scan tool used throughout the chipseq branch -- this part
doesn't use ChIP-seq or any trained model at all, just the motif itself).
Then check: do regions with a strong GATA2 motif match show more
K562-favoring (negative) ChromBPNet HepG2-vs-K562 differential scores than
regions without?

If yes, that's three independent methods (ChIP-seq-trained context models,
RNA-seq expression, and raw motif-vs-accessibility correlation) agreeing
on the same biological conclusion about GATA2 -- real convergent evidence,
not just "the pipeline ran".
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
from estfbu.motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks

cfg = load_config()

print("Loading full genome-wide ChromBPNet scan...")
df = pd.read_csv(cfg.work_dir_path("results") / "chrombpnet_full_genome_scan_hepg2_atac_peaks.csv")
print(f"  {len(df)} regions")

print("\nScoring GATA2 motif match at every region (independent of ChIP-seq/ChromBPNet)...")
ppm = trim_low_information_flanks(load_ppm("GATA2", cfg.jaspar_pfm_cache))
genome = Fasta(cfg.genome_fasta)
window = 168
half = window // 2

t0 = time.time()
gata2_scores = np.full(len(df), np.nan)
for i, row in enumerate(df.itertuples()):
    try:
        seq = genome[row.chrom][int(row.center) - half:int(row.center) + half].seq.upper()
        if len(seq) != window:
            continue
        _, v_fwd = best_match(seq, ppm)
        _, v_rev = best_match(reverse_complement(seq), ppm)
        gata2_scores[i] = max(v_fwd, v_rev)
    except Exception:
        continue
    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{len(df)} scored, {time.time()-t0:.0f}s elapsed", flush=True)

df["gata2_motif_score"] = gata2_scores
df = df.dropna(subset=["gata2_motif_score"])
print(f"  done in {time.time()-t0:.0f}s, {len(df)} regions with valid scores")

# split into "strong GATA2 motif" (top 10%) vs "rest"
threshold = df["gata2_motif_score"].quantile(0.90)
strong = df[df["gata2_motif_score"] >= threshold]
rest = df[df["gata2_motif_score"] < threshold]

print(f"\n{len(strong)} regions with strong GATA2 motif (score >= {threshold:.3f}, top 10%)")
print(f"{len(rest)} regions without")

print(f"\nMean HepG2-vs-K562 diff (positive = HepG2-favoring):")
print(f"  strong GATA2 motif regions: {strong['hepg2_vs_k562_diff'].mean():.4f}")
print(f"  other regions:              {rest['hepg2_vs_k562_diff'].mean():.4f}")

u_stat, p_value = stats.mannwhitneyu(strong["hepg2_vs_k562_diff"], rest["hepg2_vs_k562_diff"],
                                       alternative="two-sided")
print(f"\nMann-Whitney U test: U={u_stat:.0f}, p={p_value:.2e}")

direction = "K562-favoring (negative)" if strong["hepg2_vs_k562_diff"].mean() < rest["hepg2_vs_k562_diff"].mean() else "HepG2-favoring (positive)"
print(f"\nConclusion: regions with strong GATA2 motifs are shifted toward {direction} "
      f"accessibility relative to other regions.")
print("This independently uses: (1) GATA2's JASPAR motif (no ChIP-seq, no trained model), "
      "(2) ChromBPNet's genome-wide accessibility predictions (trained on ATAC-seq only, no "
      "GATA2-specific information at all). Compare against: chipseq branch says GATA2 is "
      "K562-specific (real ChIP-seq-trained comparative model); RNA-seq says GATA2 TPM is "
      "5.4 (HepG2) vs 97.2 (K562).")

out_path = cfg.work_dir_path("results") / "cross_validation_gata2_motif_vs_chrombpnet_accessibility.csv"
df.to_csv(out_path, index=False)
print(f"\nFull annotated data saved to {out_path}")
