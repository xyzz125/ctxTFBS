"""Full-scale ChromBPNet scan: ALL ATAC peaks for a given cell type,
scored with both cell types' pretrained models, for a genuine (not
bounded-demo) differential accessibility comparison between any two
configured cell types (not hardcoded to HepG2/K562 -- pass any pair with
a chrombpnet_models entry in the config).

Checkpointed in chunks -- a run this long (~9hr estimated for ~165k peaks)
needs to survive interruptions (laptop sleep, terminal disconnect) without
losing progress. Safe to kill and re-run at any point; already-completed
chunks are skipped.

Usage: python3 test_scripts/chrombpnet_full_genome_scan.py [cell_type_a] [cell_type_b]
       (defaults to HepG2 K562, the pair this repo's bundled data covers)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from estfbu.config import load_config
from estfbu import chrombpnet_scoring as cbp

CHUNK_SIZE = 2000

cell_type_a = sys.argv[1] if len(sys.argv) > 1 else "HepG2"
cell_type_b = sys.argv[2] if len(sys.argv) > 2 else "K562"
col_a, col_b = cell_type_a.lower(), cell_type_b.lower()

cfg = load_config()
ckpt_dir = cfg.work_dir_path("chrombpnet_full_scan_checkpoints") / f"{col_a}_vs_{col_b}"

ct = cfg.cell_type(cell_type_a)
regions = []
with open(ct.atac_bed) as f:
    for line in f:
        parts = line.split()
        regions.append((parts[0], (int(parts[1]) + int(parts[2])) // 2))

n_total = len(regions)
n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"{n_total} {cell_type_a} ATAC peaks, {n_chunks} chunks of {CHUNK_SIZE}, scoring against {cell_type_b}")

t_start = time.time()
for chunk_i in range(n_chunks):
    ckpt_path = ckpt_dir / f"chunk_{chunk_i:04d}.npz"
    if ckpt_path.exists():
        continue

    lo, hi = chunk_i * CHUNK_SIZE, min((chunk_i + 1) * CHUNK_SIZE, n_total)
    chunk_regions = regions[lo:hi]

    t0 = time.time()
    scores_a = cbp.score_regions_bulk(chunk_regions, cell_type_a, cfg)
    scores_b = cbp.score_regions_bulk(chunk_regions, cell_type_b, cfg)
    dt = time.time() - t0

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ckpt_path,
        chrom=np.array([r[0] for r in chunk_regions], dtype=object),
        center=np.array([r[1] for r in chunk_regions]),
        **{f"{col_a}_score": scores_a, f"{col_b}_score": scores_b},
    )

    elapsed = time.time() - t_start
    done = chunk_i + 1
    eta_min = (elapsed / done) * (n_chunks - done) / 60
    print(f"  chunk {done}/{n_chunks} done in {dt:.0f}s (elapsed {elapsed/60:.0f}min, "
          f"ETA {eta_min:.0f}min)", flush=True)

print("\nAll chunks done, assembling final CSV...")
frames = []
for chunk_i in range(n_chunks):
    d = np.load(ckpt_dir / f"chunk_{chunk_i:04d}.npz", allow_pickle=True)
    frames.append(pd.DataFrame({
        "chrom": d["chrom"], "center": d["center"],
        f"{col_a}_score": d[f"{col_a}_score"], f"{col_b}_score": d[f"{col_b}_score"],
    }))
df = pd.concat(frames, ignore_index=True)
diff_col = f"{col_a}_vs_{col_b}_diff"
df[diff_col] = df[f"{col_a}_score"] - df[f"{col_b}_score"]
df = df.dropna().sort_values(diff_col, ascending=False)

out_path = cfg.work_dir_path("results") / f"chrombpnet_full_genome_scan_{col_a}_atac_peaks.csv"
df.to_csv(out_path, index=False)
print(f"\n{len(df)} valid regions -> {out_path}")
print(f"\nTop 10 most {cell_type_a}-specific:")
print(df.head(10).to_string(index=False))
print(f"\nTop 10 most {cell_type_b}-specific:")
print(df.tail(10).to_string(index=False))
