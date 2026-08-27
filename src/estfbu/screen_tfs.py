"""Genome-wide TF screening without specifying a TF up front -- the
whiteboard's "是/否指定TF" (specify a TF or not) branch, the "no" side.

Given the full genome-wide ChromBPNet scan (no ChIP-seq, no trained
per-TF model), rank ALL 198 known TF motifs by how enriched they are in
differentially-accessible regions (specificity mode) or high-accessibility
regions (activity mode) -- same methodology validated in
test_scripts/cross_validate_gata2.py (which found GATA2 significantly
K562-enriched, agreeing with the chipseq branch and RNA-seq), generalized
from one TF to all 198 so you don't have to already know which TF to ask
about.

Checkpointed per-TF (this takes ~90min for all 198 TFs at the measured
~28s/TF rate) so it survives interruption, same pattern used for the
full genome-wide ChromBPNet scan.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy import stats

from .expression import get_tpm
from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks

def _all_tf_names(jaspar_cache_path: str):
    with open(jaspar_cache_path) as f:
        return list(json.load(f).keys())


def _score_tf_genome_wide(tf, genome, chroms, centers, ppm_cache_path, window: int):
    ppm = trim_low_information_flanks(load_ppm(tf, ppm_cache_path))
    half = window // 2
    scores = np.full(len(chroms), np.nan)
    for i, (chrom, center) in enumerate(zip(chroms, centers)):
        try:
            seq = genome[chrom][int(center) - half:int(center) + half].seq.upper()
        except Exception:
            continue
        if len(seq) != window:
            continue
        _, v_fwd = best_match(seq, ppm)
        _, v_rev = best_match(reverse_complement(seq), ppm)
        scores[i] = max(v_fwd, v_rev)
    return scores


def screen_all_tfs(cfg, cell_types, scan_csv_path=None, mode="specificity",
                    top_quantile=0.9, checkpoint_dir=None, progress_every=10):
    """cell_types: [A] for mode='activity' (ranks TFs by whether their
    motif is enriched in high-accessibility regions for A alone -- needs a
    scan with an '{a}_score' column), or [A, B] for mode='specificity'
    (ranks TFs by whether their motif is enriched in strongly differential
    A-vs-B regions -- needs a scan with an '{a}_vs_{b}_diff' column). Not
    hardcoded to any specific pair -- works for whatever two cell types the
    scan CSV was actually built for (see chrombpnet_full_genome_scan.py,
    which also takes any pair as of this fix), matching whatever the user
    entered for this run.

    Returns a DataFrame, one row per TF, sorted by signed effect size
    (mean_diff_strong - mean_diff_rest) descending -- NOT by p-value. The
    underlying test is two-sided (is motif enrichment associated with a
    *different* target value, either way), so two TFs pointing in opposite
    directions can get the identical p-value; sorting by p-value alone
    silently interleaves both directions (a real bug this fixes -- rank 1
    used to come out K562-favoring on the validated HepG2-vs-K562 scan
    despite the screen's whole point being to find HepG2-specific TFs).
    Sorting by signed effect size instead puts cell_types[0]-favoring TFs
    at the top (strongest first), cell_types[1]-favoring at the bottom,
    with effect_direction and pvalue both still available as columns for
    the caller to filter/inspect further. Columns: tf, mean_diff_strong,
    mean_diff_rest, effect_direction, pvalue, n_strong, tpm_a, tpm_b. The
    TPM columns are what the user checks to confirm a top-ranked TF is
    actually expressed in the cell type it's being suggested for -- a
    motif being enriched doesn't guarantee that on its own. tpm_b is NaN
    in mode='activity' (only one cell type is relevant there).
    """
    if mode == "specificity":
        if len(cell_types) != 2:
            raise ValueError("mode='specificity' requires exactly 2 cell_types")
        a, b = cell_types
        target_col = f"{a.lower()}_vs_{b.lower()}_diff"
    else:
        a = cell_types[0]
        b = cell_types[1] if len(cell_types) > 1 else None
        target_col = f"{a.lower()}_score"

    scan_csv_path = scan_csv_path or (cfg.work_dir_path("results") / f"chrombpnet_full_genome_scan_{a.lower()}_atac_peaks.csv")
    scan_csv_path = Path(scan_csv_path)

    # If the scan doesn't exist yet, but a precomputed one is configured
    # and actually has the column this call needs, copy it into place --
    # this used to live only in run_pipeline.py's interactive script, so
    # the CLI (cli.py) would hit "[STOP] ... not found" and tell you to
    # run a multi-hour scan even when the bundled precomputed one (which
    # covers exactly this pair) was sitting right there. Column-checked,
    # not a hardcoded "only if cell_types[0] == 'HepG2'" guard, so this
    # works for whatever pair the precomputed file actually covers.
    precomputed = getattr(cfg, "precomputed_genome_scan", None)
    if not scan_csv_path.exists() and precomputed and Path(precomputed).exists():
        precomputed_cols = pd.read_csv(precomputed, nrows=0).columns
        if target_col in precomputed_cols:
            import shutil
            scan_csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(precomputed, scan_csv_path)

    df = pd.read_csv(scan_csv_path)
    if target_col not in df.columns:
        raise ValueError(f"scan CSV missing expected column '{target_col}' for mode='{mode}' "
                          f"cell_types={cell_types} (available columns: {list(df.columns)}) -- "
                          f"this scan CSV wasn't built for this cell-type pair; build one with "
                          f"chrombpnet_full_genome_scan.py first")

    genome = Fasta(cfg.genome_fasta)
    chroms = df["chrom"].to_numpy()
    centers = df["center"].to_numpy()
    target = df[target_col].to_numpy()

    # keyed on mode + cell_types, not just tf name -- a bare {tf}.json
    # checkpoint would silently return a previous run's cached row for a
    # DIFFERENT mode/cell-type pair on the same TF (e.g. running
    # mode='activity' after mode='specificity' would load the specificity
    # numbers back as if they were the activity result)
    ckpt_dir = (checkpoint_dir or cfg.work_dir_path("tf_screen_checkpoints")) / f"{mode}_{'_'.join(cell_types)}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tf_names = _all_tf_names(cfg.jaspar_pfm_cache)

    results = []
    t0 = time.time()
    for i, tf in enumerate(tf_names):
        ckpt_path = ckpt_dir / f"{tf}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                results.append(json.load(f))
            continue

        # RNA-seq TPM in each relevant cell type -- a motif being enriched
        # or a region being accessible doesn't guarantee the TF's own gene
        # is actually expressed there, so this is surfaced on every row
        # (not just the eventual top candidates) for the user to check
        # before trusting the ranking.
        tpm_a = get_tpm(tf, a, cfg)
        tpm_b = get_tpm(tf, b, cfg) if b else float("nan")

        motif_scores = _score_tf_genome_wide(tf, genome, chroms, centers, cfg.jaspar_pfm_cache, cfg.model.seq_len)
        valid = ~np.isnan(motif_scores)
        if valid.sum() < 100:
            row = {"tf": tf, "mean_diff_strong": None, "mean_diff_rest": None,
                   "pvalue": None, "n_strong": 0, "error": "too few valid regions",
                   "tpm_a": tpm_a, "tpm_b": tpm_b}
        else:
            threshold = np.nanquantile(motif_scores, top_quantile)
            strong_mask = valid & (motif_scores >= threshold)
            rest_mask = valid & (motif_scores < threshold)
            strong_vals = target[strong_mask]
            rest_vals = target[rest_mask]
            if len(strong_vals) < 20 or len(rest_vals) < 20:
                row = {"tf": tf, "mean_diff_strong": None, "mean_diff_rest": None,
                       "pvalue": None, "n_strong": int(strong_mask.sum()), "error": "too few in one group",
                       "effect_direction": None, "tpm_a": tpm_a, "tpm_b": tpm_b}
            else:
                u_stat, pvalue = stats.mannwhitneyu(strong_vals, rest_vals, alternative="two-sided")
                mean_strong, mean_rest = float(np.mean(strong_vals)), float(np.mean(rest_vals))
                # the Mann-Whitney test above is two-sided (is motif
                # enrichment associated with a *different* target value,
                # either way) -- it says nothing about which way. A TF
                # whose motif correlates with LOWER target (e.g. more
                # b-favoring, if target=a_vs_b_diff) gets exactly the same
                # p-value as one correlating with higher target, so ranking
                # by p-value alone silently mixes both directions together.
                # This column is what lets a caller filter to the direction
                # they actually asked about instead of assuming the top of
                # the p-value-sorted list is automatically the direction
                # they wanted.
                if mode == "specificity":
                    effect_direction = a if mean_strong > mean_rest else b
                else:
                    effect_direction = "high_accessibility" if mean_strong > mean_rest else "low_accessibility"
                row = {
                    "tf": tf,
                    "mean_diff_strong": mean_strong,
                    "mean_diff_rest": mean_rest,
                    "pvalue": float(pvalue),
                    "n_strong": int(strong_mask.sum()),
                    "error": None,
                    "effect_direction": effect_direction,
                    "tpm_a": tpm_a,
                    "tpm_b": tpm_b,
                }
        with open(ckpt_path, "w") as f:
            json.dump(row, f)
        results.append(row)

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            eta_min = (elapsed / (i + 1)) * (len(tf_names) - i - 1) / 60
            print(f"  [{i+1}/{len(tf_names)}] {tf} done, {elapsed:.0f}s elapsed, ETA {eta_min:.0f}min", flush=True)

    result_df = pd.DataFrame(results)
    valid_df = result_df[result_df["pvalue"].notna()].copy()
    valid_df["_effect_size"] = valid_df["mean_diff_strong"] - valid_df["mean_diff_rest"]
    valid_df = valid_df.sort_values("_effect_size", ascending=False).drop(columns="_effect_size")
    invalid_df = result_df[result_df["pvalue"].isna()]
    return pd.concat([valid_df, invalid_df], ignore_index=True)
