"""Demo/validation of the no-ChIP-seq branch: score a subset of HepG2 and
K562 ATAC peaks with their respective pretrained ChromBPNet models, and
compute a HepG2-vs-K562 specificity score per region -- this is the data
a bubble/dot plot (x=region or TF, y=cell type, size=score) would consume.

NOT a full genome-wide scan (that's ~4.5hr/cell-type at the measured rate on
this machine, see README) -- this samples 5000 regions per cell type as a
bounded, realistic demonstration that the ChromBPNet scoring path works
end-to-end on real ENCODE data.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from estfbu.config import load_config
from estfbu import chrombpnet_scoring as cbp

N_REGIONS = 5000

cfg = load_config()

t0 = time.time()
print(f"Scanning {N_REGIONS} HepG2 ATAC peaks...")
regions, hepg2_scores = cbp.scan_atac_peaks("HepG2", cfg, max_regions=N_REGIONS)
print(f"  done in {time.time()-t0:.0f}s")

t0 = time.time()
print(f"Scoring the SAME {N_REGIONS} regions with the K562 model (for direct comparison)...")
k562_scores = cbp.score_regions_bulk(regions, "K562", cfg)
print(f"  done in {time.time()-t0:.0f}s")

df = pd.DataFrame({
    "chrom": [r[0] for r in regions],
    "center": [r[1] for r in regions],
    "hepg2_score": hepg2_scores,
    "k562_score": k562_scores,
})
df["hepg2_vs_k562_diff"] = df["hepg2_score"] - df["k562_score"]
df = df.dropna()
df = df.sort_values("hepg2_vs_k562_diff", ascending=False)

out_path = cfg.work_dir_path("results") / "chrombpnet_demo_scan_hepg2_atac_peaks.csv"
df.to_csv(out_path, index=False)

print(f"\n{len(df)} valid regions scored, saved to {out_path}")
print(f"\nTop 5 most HepG2-specific regions (by ChromBPNet accessibility score):")
print(df.head(5).to_string(index=False))
print(f"\nTop 5 most K562-specific regions (among HepG2 ATAC peaks -- i.e. least HepG2-specific):")
print(df.tail(5).to_string(index=False))
