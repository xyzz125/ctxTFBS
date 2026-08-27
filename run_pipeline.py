#!/usr/bin/env python3
"""esTFBU -- interactive, step-by-step driver for the pipeline in
src/estfbu/.

At every step it asks only the detail(s) that step needs, runs it, shows
you the real output, and asks whether to continue or stop there so you
can go inspect it.

    python3 run_pipeline.py

This is a thin driver: all of the actual pipeline logic (config loading,
motif scanning, data prep, model training, genetic-algorithm design,
ChromBPNet scoring, TF annotation, post-processing) lives in
src/estfbu/, the single source of truth also used by
test_scripts/test_regression.py (17 regression tests) and
`python -m estfbu.cli ...` for non-interactive/scripted runs. This file
used to duplicate that logic instead of importing it (kept as one
standalone file with no import indirection); that duplication was a real
maintenance-drift risk -- a fix applied to one copy could silently miss
the other -- so it now imports the package like everything else does.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta, FastaNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from estfbu import blacklist, chrombpnet_scoring, data_prep, design, motif_hits, postprocess, screen_tfs, train
from estfbu.config import DEFAULT_CONFIG_PATH, QUICKSTART_CONFIG_PATH, load_config, seed_everything
from estfbu.motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks
from estfbu.oligo_library import emit_oligo_library
from estfbu.reporting import write_bubble_plot, write_gc_comparison_plot, write_step_manifest


# =============================================================================
# INTERACTIVE DRIVER -- each step asks only what it needs, runs it, prints
# real output, then asks once whether to continue or stop and inspect.
# =============================================================================

def ask(prompt, default=None, choices=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            raw = default
        if not raw:
            print("  (required, try again)")
            continue
        if choices and raw not in choices:
            print(f"  must be one of {choices}")
            continue
        return raw


def ask_yes_no(prompt, default=False):
    d = "y" if default else "n"
    raw = input(f"{prompt} [y/n, default {d}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def checkpoint(label, detail=""):
    print(f"\n--- checkpoint: {label} ---")
    if detail:
        print(detail)
    resp = input("Press Enter to continue to the next stage, or 's' to stop here: ").strip().lower()
    if resp == "s":
        print("\nStopped by request. Everything printed above (file paths, previews) is real "
              "output already written to disk -- open those files directly to inspect them. "
              "Re-run this script to continue from the top.")
        sys.exit(0)


def ask_oligo_library_params(cfg):
    """Asks once per design run whether to also emit an esMPRA-shaped
    oligo library from the designed sequences (see oligo_library.py's
    docstring for the validation caveat), and if so, the parameters it
    needs. Returns a kwargs dict for emit_oligo_library, or None."""
    if not ask_yes_no("Also emit an esMPRA-shaped oligo library from this design? (trims/tiles to "
                       "fit array synthesis, adds adapters + barcode + scrambled controls -- see "
                       "emit_oligo_library's docstring for the validation caveat)", default=False):
        return None
    print("  (all three of oligo_pre/oligo_after/barcode_pre are required -- no default is assumed "
          "silently here, since a library with no real adapters isn't a real esMPRA-shaped library, "
          "it just looks like one. Not independently verified in this environment, but a review pass "
          "reported esMPRA's own published adapters as GGCCGCTTGACG (oligo_pre) / CACTGCGGCTCC "
          "(oligo_after) / CGAACCTCTAGA (barcode_pre), and its own barcode length as 20 (vs. this "
          "prompt's own default of 12) -- worth trying if you don't have your own.)")
    return {
        "oligo_pre": ask("5' adapter/primer to prepend"),
        "oligo_after": ask("3' adapter/primer to append"),
        "barcode_pre": ask("Fixed spacer right before the random barcode"),
        "barcode_length": int(ask("Random barcode length", default="12")),
        "max_oligo_len": int(ask("Total oligo length budget incl. adapters+barcode", default="230")),
        "seed": cfg.training.random_seed,
        "restriction_sites": cfg.postprocess.restriction_sites,
    }


def ask_tf(prompt, valid_tfs, extra_note=""):
    """Prompts for a TF name, re-asking until it's one we can actually
    run with -- instead of failing several steps later with a
    FileNotFoundError deep inside data prep. `valid_tfs` should already
    have blacklisted TFs excluded (see blacklist.exclude_blacklisted) so a
    blocked TF is never even offered as a choice."""
    print(f"  {len(valid_tfs)} TF(s) available: {', '.join(valid_tfs) if valid_tfs else '(none)'}")
    if extra_note:
        print(f"  {extra_note}")
    if not valid_tfs:
        print("  [STOP] no TF has the data this run needs.")
        sys.exit(1)
    while True:
        tf = ask(prompt)
        if tf in valid_tfs:
            return tf
        print(f"  '{tf}' isn't in the list above (check spelling/case) -- try again")


def run_chipseq_screen(valid_tfs, cell_types, cfg):
    """The has-ChIP-seq branch's 'no TF specified' path: ranks every
    ChIP-seq-available (and non-blacklisted) TF by its own trained/
    pretrained model's confidence on its own held-out (not full, not
    in-sample) ChIP-seq-derived positive windows, returns the best 1-3,
    and (for specificity, 2 cell types) a bubble plot -- the has-ChIP-seq
    equivalent of the no-ChIP-seq branch's 'screen' action. Unlike that
    branch's screen (pure motif math, always fast), this one needs a real
    trained model per candidate TF per cell type -- slow for any TF/
    cell-type pairing that isn't pretrained (see the from-scratch-training
    warning already printed by the caller)."""
    target = ask("Target", default="specificity", choices=["specificity", "activity"])
    if target == "specificity" and len(cell_types) != 2:
        print("\n[STOP] target=specificity requires exactly 2 cell types")
        sys.exit(1)
    n_top = int(ask("How many top candidates to keep ('best 1' or 'best 3' style)", default="3"))

    print(f"\n=== [2/2] Scoring {len(valid_tfs)} candidate TF(s) for target={target} ===")
    print("  (scored on each model's held-out 10% test split, not its full positive set --")
    print("   a full-positive-set score would be in-sample for any pretrained TF)")
    t0 = time.time()
    results = train.rank_chipseq_tfs(cfg, valid_tfs, cell_types, target)
    for r in results:
        if target == "specificity":
            print(f"  {r['tf']}: {cell_types[0]}={r['score_a']:.4f} auc={r['auc_a']:.3f} (tpm={r['tpm_a']:.1f}), "
                  f"{cell_types[1]}={r['score_b']:.4f} auc={r['auc_b']:.3f} (tpm={r['tpm_b']:.1f}), "
                  f"diff={r['score_a'] - r['score_b']:+.4f}")
        else:
            print(f"  {r['tf']}: {cell_types[0]}={r['score_a']:.4f} auc={r['auc_a']:.3f} (tpm={r['tpm_a']:.1f})")

    top = results[:n_top]
    print(f"\n  Top {n_top}: {[r['tf'] for r in top]}")
    print("  (tpm columns above are real RNA-seq expression -- a high model score doesn't prove a top "
          "candidate's own gene is actually expressed in that cell type; check before trusting it)")

    out_csv = cfg.work_path("results", f"chipseq_screen_{target}_{'_'.join(cell_types)}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    write_step_manifest(cfg, "screen", f"chipseq_{target}_{'_'.join(cell_types)}", out_csv, "csv",
                         len(results), cfg.training.random_seed, time.time() - t0)
    print(f"  {out_csv}")

    if target == "specificity":
        out_bubble = cfg.work_path("results", f"chipseq_screen_{target}_{'_'.join(cell_types)}_bubble.png")
        write_bubble_plot([r["tf"] for r in top], [r["score_a"] for r in top], [r["score_b"] for r in top],
                           cell_types, cfg, out_bubble,
                           f"Top {n_top} candidate TFs: bubble size = each TF's own trained-model\n"
                           f"score on its real ChIP-seq positives, color = RNA-seq expression")
        print(f"  {out_bubble}")
    else:
        print("  (bubble plot needs target=specificity with exactly 2 cell types -- skipped)")


def run_chipseq_interactive(cfg):
    cell_types = [c.strip() for c in ask("Cell types, comma-separated (e.g. HepG2,K562)").split(",") if c.strip()]

    unpretrained = [ct for ct in cell_types if not cfg.cell_type(ct).pretrained]
    if unpretrained:
        device_note = ("MPS (Apple Silicon GPU) will be used automatically -- measured ~20-25 "
                        "min per model on real data" if torch.backends.mps.is_available() else
                        "no MPS available, will run on CPU -- measured ~70+ min PER EPOCH on CPU "
                        "alone; a GPU is strongly recommended (see README's System Requirements)")
        print(f"\n  NOTE: {unpretrained} has no pretrained weights -- this run will train a "
              f"model from scratch. {device_note}. If target=specificity, TWO models train "
              f"sequentially (double the wait, ~35-40 min total measured with MPS). Consider "
              f"Ctrl-C now if you don't want to commit to that.")

    print(f"\n=== [1/4] Blacklist + ChIP-seq availability for {cell_types} ===")
    valid_tfs = blacklist.exclude_blacklisted(blacklist.available_chipseq_tfs(cell_types, cfg), cfg)
    print(f"  {len(valid_tfs)} TF(s) available: {', '.join(valid_tfs) if valid_tfs else '(none)'}")
    if not valid_tfs:
        print("  [STOP] no TF has the data this run needs.")
        sys.exit(1)

    if not ask_yes_no("Specify one TF to design for? ('n' instead ranks all available TFs and "
                       "returns the best 1-3 candidates)", default=True):
        run_chipseq_screen(valid_tfs, cell_types, cfg)
        return

    tf = ask_tf("Transcription factor", valid_tfs,
                extra_note=f"(has-ChIP-seq branch needs a bundled ChIP-seq bed file per cell type, "
                           f"AND the TF must not be in {cfg.blacklist_file} -- this list is already "
                           f"filtered on both, so anything you pick here is guaranteed runnable)")
    checkpoint("TF selection", f"  '{tf}' has ChIP-seq data for {cell_types} and is not blacklisted")

    print(f"\n=== [2/4] Data preparation for {cell_types} ===")
    h5_paths = {}
    for ct in cell_types:
        h5_paths[ct] = data_prep.prepare(tf, ct, cfg)
        print(f"  {ct}: {h5_paths[ct]}")
    checkpoint("data preparation", "  These .h5 files hold the masked-core-motif training windows "
                                    "used to train/reuse the TFBS-context model below.")

    starting_sequence = None
    if ask_yes_no("Seed the design around a fixed starting sequence instead of de novo?"):
        starting_sequence = ask(f"Starting sequence (exactly {cfg.model.seq_len}bp; optionally mark "
                                 f"positions to keep fixed with lowercase letters, e.g. a known-important "
                                 f"element you don't want the GA mutating -- uppercase stays free to optimize)")
    if ask_yes_no(f"Full GA run is {cfg.genetic_algorithm.n_iterations} iterations "
                  f"(~{cfg.genetic_algorithm.n_iterations * 0.32:.0f} min per objective). Run a "
                  f"quick demo with fewer iterations instead?", default=True):
        cfg.genetic_algorithm.n_iterations = int(ask("How many iterations for the demo", default="20"))
    gc_target = float(ask("Target GC content for final filtering", default="0.5"))

    # No upfront activity-vs-specificity question: the GA needs one fitness
    # objective per run (can't optimize both at once), but you shouldn't
    # have to guess which before seeing either -- run both and hand back
    # both outputs plus a comparison plot. Only "activity" is meaningful
    # with a single cell type (specificity needs two to compare).
    objectives = ["activity", "specificity"] if len(cell_types) == 2 else ["activity"]
    if len(cell_types) == 1:
        print(f"\n  (only 1 cell type given -- specificity needs 2 to compare, running activity only)")
    oligo_params = ask_oligo_library_params(cfg)

    print(f"\n=== [3/4] Genetic algorithm design ({cfg.genetic_algorithm.n_iterations} "
          f"iterations x {len(objectives)} objective(s)) ===")
    final_by_target = {}
    out_fasta_by_target = {}
    for target in objectives:
        t0 = time.time()
        seqs, scores = design.run_ga(tf, cell_types, target, cfg, starting_sequence=starting_sequence)
        print(f"  [{target}] {len(seqs)} unique candidates, best raw score={scores.max():.4f}")

        final = postprocess.filter_and_dedup(tf, seqs, scores, cfg, target_gc=gc_target)
        print(f"  [{target}] {len(final)} sequences survived filtering/dedup (target GC={gc_target})")
        final_by_target[target] = final

        out_fasta = cfg.work_path("results", f"{tf}_{'_'.join(cell_types)}_{target}.fasta")
        with open(out_fasta, "w") as f:
            for i, seq in enumerate(final):
                f.write(f">{tf}_{target}_{i}\n{seq}\n")
        write_step_manifest(cfg, "design", f"{tf}_{'_'.join(cell_types)}_{target}", out_fasta, "fasta",
                             len(final), cfg.training.random_seed, time.time() - t0)
        out_fasta_by_target[target] = out_fasta
        if oligo_params:
            inserts_out, oligos_out, manifest_out = emit_oligo_library(
                out_fasta, cfg.work_dir_path("results"), **oligo_params)
            print(f"  [{target}] oligo library: {oligos_out} (inserts for esMPRA's --ref_fa: {inserts_out})")

    print(f"\n=== [4/4] Done ===")
    for target, path in out_fasta_by_target.items():
        print(f"  {target}: {path} ({len(final_by_target[target])} designed sequences)")

    if len(objectives) == 2:
        out_plot = cfg.work_path("results", f"{tf}_{'_'.join(cell_types)}_activity_vs_specificity_gc.png")
        write_gc_comparison_plot(final_by_target["activity"], "Activity", final_by_target["specificity"],
                                  "Specificity", out_plot, f"{tf}: activity vs specificity design, GC content")
        print(f"  comparison plot: {out_plot}")


def run_chrombpnet_interactive(cfg):
    # TF question first, same shape as the has-ChIP-seq branch. A specified
    # TF means something for both 'score' (re-ranks scanned regions by that
    # TF's own motif match) and 'design' (run_ga_chrombpnet switches its
    # fitness function to a gradient-based motif-importance proxy -- see
    # that function's docstring -- instead of raw predicted accessibility,
    # making a TF-specific design possible without ChIP-seq). 'screen' is
    # explicitly the "don't specify a TF" mode (it's what finds candidate
    # TFs in the first place), so it's only offered when no TF is given.
    print("\n=== [1] Specify a TF? ===")
    if ask_yes_no("Specify a TF? ('n' also lets you pick a genome-wide screen instead; "
                   "'y' restricts you to score or design, the two modes a TF can affect)", default=False):
        valid_tfs = blacklist.exclude_blacklisted(screen_tfs._all_tf_names(cfg.jaspar_pfm_cache), cfg)
        tf = ask_tf("Transcription factor", valid_tfs,
                    extra_note=f"(any known JASPAR motif works, as long as it's not in "
                               f"{cfg.blacklist_file} -- showing all {len(valid_tfs)} that qualify)")
        checkpoint("TF selection", f"  '{tf}' is a known JASPAR motif and is not blacklisted")
        action = ask("Action", default="score", choices=["score", "design"])
    else:
        tf = None
        action = ask("Action", default="score", choices=["score", "design", "screen"])

    cell_types = [c.strip() for c in ask("Cell types, comma-separated (e.g. HepG2,K562)").split(",") if c.strip()]

    if action == "screen":
        # no starting-sequence concept for a genome-wide screen (nothing is
        # being designed/seeded) -- target is the only remaining question
        target = ask("Target", default="specificity", choices=["specificity", "activity"])
        if target == "specificity" and len(cell_types) != 2:
            print("\n[STOP] target=specificity requires exactly 2 cell types")
            sys.exit(1)
        # screen_all_tfs itself copies the bundled precomputed scan into
        # place if it covers this cell-type pair (column-checked), so
        # nothing needs duplicating here -- just report what happened.
        scan_csv = cfg.work_dir_path("results") / f"chrombpnet_full_genome_scan_{cell_types[0].lower()}_atac_peaks.csv"
        scan_csv_existed = scan_csv.exists()
        print(f"\n=== [2] Genome-wide screen: ranking all 198 known TFs ===")
        t0 = time.time()
        try:
            result_df = screen_tfs.screen_all_tfs(cfg, cell_types, scan_csv_path=scan_csv, mode=target)
        except ValueError as e:
            print(f"\n[STOP] {e}\n(needs a full genome-wide ChromBPNet scan for {cell_types} to "
                  f"already exist -- see test_scripts/chrombpnet_full_genome_scan.py <cell_type_a> "
                  f"<cell_type_b> -- this takes several hours, so it isn't triggered automatically here; "
                  f"the bundled precomputed scan only covers HepG2 vs K562).")
            sys.exit(1)
        if not scan_csv_existed and scan_csv.exists():
            print(f"  (copied bundled precomputed scan into place -- no multi-hour scan needed)")
        print(result_df.head(10).to_string(index=False))
        print("  (tpm_a/tpm_b above are real RNA-seq expression -- motif enrichment alone doesn't "
              "prove a top candidate's own gene is actually expressed in that cell type; check before trusting it)")
        out_csv = cfg.work_path("results", f"tf_screen_{target}_{'_'.join(cell_types)}.csv")
        result_df.to_csv(out_csv, index=False)
        write_step_manifest(cfg, "screen", f"chrombpnet_{target}_{'_'.join(cell_types)}", out_csv, "csv",
                             len(result_df), cfg.training.random_seed, time.time() - t0)
        print(f"\n=== Done: {out_csv} ({len(result_df)} TFs ranked) ===")

        if target == "specificity" and len(cell_types) == 2:
            n_top = int(ask("How many top candidates to bubble-plot ('best 1' or 'best 3' style)", default="3"))
            print(f"\n=== [3] Bubble plot: top {n_top} candidates ===")
            top_tfs = result_df[result_df["pvalue"].notna()].head(n_top)["tf"].tolist()
            genome = Fasta(cfg.genome_fasta)
            scan_df = pd.read_csv(scan_csv)
            chroms, centers = scan_df["chrom"].to_numpy(), scan_df["center"].to_numpy()
            scores_a, scores_b = [], []
            for tf in top_tfs:
                motif_scores = screen_tfs._score_tf_genome_wide(tf, genome, chroms, centers, cfg.jaspar_pfm_cache, cfg.model.seq_len)
                valid = ~np.isnan(motif_scores)
                threshold = np.nanquantile(motif_scores, 0.9)
                strong = valid & (motif_scores >= threshold)
                scores_a.append(scan_df[f"{cell_types[0].lower()}_score"][strong].mean())
                scores_b.append(scan_df[f"{cell_types[1].lower()}_score"][strong].mean())
            out_bubble = cfg.work_path("results", f"tf_screen_{target}_{'_'.join(cell_types)}_bubble.png")
            write_bubble_plot(top_tfs, scores_a, scores_b, cell_types, cfg, out_bubble,
                               f"Top {n_top} candidate TFs: bubble size = ChromBPNet accessibility\n"
                               f"in that TF's strong-motif regions, color = RNA-seq expression")
            print(f"  {out_bubble}")
        else:
            print("\n  (bubble plot skipped -- needs target=specificity with exactly 2 cell types)")
        return

    if action == "design":
        starting_sequence = None
        auto_fix_top_fraction = None
        if ask_yes_no("Seed the design around a fixed starting sequence instead of de novo?"):
            starting_sequence = ask("Starting sequence (exactly 2114bp; optionally mark positions to "
                                     "keep fixed with lowercase letters -- uppercase stays free to optimize)")
            if ask_yes_no("Also auto-detect and protect the most important existing positions "
                          "(by gradient importance, no manual marking needed -- unioned with any "
                          "lowercase marks above, not a replacement for them)?", default=False):
                auto_fix_top_fraction = float(ask("Fraction of positions to auto-protect", default="0.05"))
        if ask_yes_no(f"Full GA run is {cfg.genetic_algorithm.n_iterations} iterations per "
                      f"objective. Run a quick demo with fewer instead?", default=True):
            cfg.genetic_algorithm.n_iterations = int(ask("How many iterations for the demo", default="20"))

        # No upfront activity-vs-specificity question here either, same
        # reasoning as the has-ChIP-seq branch's design path: the GA needs
        # one objective per run, so run both and compare instead of
        # guessing which one you want first.
        objectives = ["activity", "specificity"] if len(cell_types) == 2 else ["activity"]
        if len(cell_types) == 1:
            print(f"\n  (only 1 cell type given -- specificity needs 2 to compare, running activity only)")
        oligo_params = ask_oligo_library_params(cfg)

        print(f"\n=== [2] ChromBPNet-driven GA design ({cfg.genetic_algorithm.n_iterations} "
              f"iterations x {len(objectives)} objective(s)) ===")
        final_by_target = {}
        out_fasta_by_target = {}
        name_prefix = f"chrombpnet_design_{tf}_{'_'.join(cell_types)}" if tf else \
                      f"chrombpnet_design_{'_'.join(cell_types)}"
        if tf:
            print(f"  '{tf}' specified -- fitness is gradient-based motif-importance in/around "
                  f"its own PWM match, not raw accessibility (see run_ga_chrombpnet's docstring "
                  f"for why this stands in for the ChIP-seq signal this branch doesn't have)")
        for target in objectives:
            t0 = time.time()
            seqs, scores = design.run_ga_chrombpnet(cell_types, target, cfg,
                                                      starting_sequence=starting_sequence, tf=tf,
                                                      auto_fix_top_fraction=auto_fix_top_fraction)
            print(f"  [{target}] {len(seqs)} unique candidates, best raw score={scores.max():.4f}")

            final = postprocess.dedup_by_edit_distance(seqs, scores, cfg, check_restriction_sites=True)
            print(f"  [{target}] {len(final)} sequences after dedup")
            final_by_target[target] = final

            out_fasta = cfg.work_path("results", f"{name_prefix}_{target}.fasta")
            with open(out_fasta, "w") as f:
                for i, seq in enumerate(final):
                    f.write(f">chrombpnet_{target}_{i}\n{seq}\n")
            write_step_manifest(cfg, "design", f"{name_prefix}_{target}", out_fasta, "fasta",
                                 len(final), cfg.training.random_seed, time.time() - t0)
            out_fasta_by_target[target] = out_fasta
            if oligo_params:
                inserts_out, oligos_out, manifest_out = emit_oligo_library(
                    out_fasta, cfg.work_dir_path("results"), **oligo_params)
                print(f"  [{target}] oligo library: {oligos_out} (inserts for esMPRA's --ref_fa: {inserts_out})")

        print(f"\n=== Done ===")
        for target, path in out_fasta_by_target.items():
            print(f"  {target}: {path} ({len(final_by_target[target])} sequences)")

        if len(objectives) == 2:
            out_plot = cfg.work_path("results", f"{name_prefix}_activity_vs_specificity_gc.png")
            write_gc_comparison_plot(final_by_target["activity"], "Activity", final_by_target["specificity"],
                                      "Specificity", out_plot, "ChromBPNet design: activity vs specificity, GC content")
            print(f"  comparison plot: {out_plot}")
        return

    # action == "score" -- no starting-sequence concept (nothing is being
    # designed/seeded, only existing ATAC peaks are scored), and no target
    # question either: scoring a real region against a pretrained model is
    # cheap, so with 2 cell types this always computes both scores + the
    # diff in one pass rather than making you pre-pick which one you want.
    t0 = time.time()
    max_regions = int(ask("Max ATAC regions to scan (full genome-wide is ~170k regions, several "
                           "hours at the measured per-cell-type scan rate)", default="5000"))
    print(f"\n=== [2] Scanning {max_regions} {cell_types[0]} ATAC peaks with the {cell_types[0]} model ===")
    regions, primary_scores = chrombpnet_scoring.scan_atac_peaks(cell_types[0], cfg, max_regions=max_regions)
    df = pd.DataFrame({"chrom": [r[0] for r in regions], "center": [r[1] for r in regions],
                        f"{cell_types[0]}_score": primary_scores})
    print(f"  scored {len(df)} regions")
    print(df.head(5).to_string(index=False))
    checkpoint("primary scan", "")

    if len(cell_types) == 2:
        print(f"\n=== [3] Scoring the same regions with the {cell_types[1]} model ===")
        secondary_scores = chrombpnet_scoring.score_regions_bulk(regions, cell_types[1], cfg)
        df[f"{cell_types[1]}_score"] = secondary_scores
        df["diff"] = df[f"{cell_types[0]}_score"] - df[f"{cell_types[1]}_score"]
        df = df.dropna().sort_values("diff", ascending=False)
        print(df.head(5).to_string(index=False))
        checkpoint("differential scan", "  'diff' > 0 favors the first cell type, < 0 favors the second -- "
                                          f"{cell_types[0]}_score and {cell_types[1]}_score are both kept too")
        rank_col = "diff"
    else:
        df = df.dropna().sort_values(f"{cell_types[0]}_score", ascending=False)
        rank_col = f"{cell_types[0]}_score"

    if tf:
        print(f"\n=== [3.5] Restricting to '{tf}': re-ranking regions by its own motif match ===")
        ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
        genome = Fasta(cfg.genome_fasta)
        window = cfg.model.seq_len
        half = window // 2
        motif_scores = []
        for row in df.itertuples():
            try:
                seq = genome[row.chrom][int(row.center) - half:int(row.center) + half].seq.upper()
            except Exception:
                motif_scores.append(float("nan"))
                continue
            if len(seq) != window:
                motif_scores.append(float("nan"))
                continue
            _, v_fwd = best_match(seq, ppm)
            _, v_rev = best_match(reverse_complement(seq), ppm)
            motif_scores.append(max(v_fwd, v_rev))
        df[f"{tf}_motif_score"] = motif_scores
        df = df.dropna(subset=[f"{tf}_motif_score"]).sort_values(f"{tf}_motif_score", ascending=False)
        print(f"  {len(df)} regions re-ranked by {tf} motif match strength (was ranked by {rank_col})")
        print(df.head(5).to_string(index=False))
        checkpoint("TF-restricted ranking", "")

    annotate_top = int(ask("Top/bottom N regions to annotate with candidate TF matches", default="10"))
    print(f"\n=== [4] Annotating top/bottom {annotate_top} regions with TF motif matches ===")
    print("  (slow step: motif-vs-shuffled-background significance testing)")
    subset = pd.concat([df.head(annotate_top), df.tail(annotate_top)])
    annotated = motif_hits.annotate_regions(subset, cfg, top_n=3, cell_types=cell_types)
    print(annotated.head(5).to_string(index=False))

    out_csv = cfg.work_path("results", f"chrombpnet_{'_'.join(cell_types)}.csv")
    df.to_csv(out_csv, index=False)
    out_annotated_csv = cfg.work_path("results", f"chrombpnet_{'_'.join(cell_types)}_annotated_top{annotate_top}.csv")
    annotated.to_csv(out_annotated_csv, index=False)
    write_step_manifest(cfg, "score", f"chrombpnet_{'_'.join(cell_types)}", out_csv, "csv",
                         len(df), cfg.training.random_seed, time.time() - t0)
    print(f"\n=== Done: {out_csv} (full scan, {rank_col} = default sort), {out_annotated_csv} (annotated top/bottom) ===")


def main():
    print("=== esTFBU (interactive, step-by-step) ===")
    print("Two branches: 'chipseq' (needs ChIP-seq data + a per-TF trained model)")
    print("or 'chrombpnet' (only needs ATAC-seq + a pretrained accessibility model).")
    print("Each stage asks only what it needs, runs it, prints real output, then asks")
    print("once whether to continue -- answer 's' at any pause to stop and inspect.\n")

    use_quickstart = ask_yes_no(
        "Use bundled quickstart data (sample_data/, real but small -- runs immediately, "
        "no downloads except the genome)? 'n' uses your own config/default_config.yaml "
        "(full-scale, needs all external data set up per README)", default=True)
    config_path = QUICKSTART_CONFIG_PATH if use_quickstart else None
    cfg = load_config(config_path)
    seed_everything(cfg.training.random_seed)
    print(f"  seeded RNGs with {cfg.training.random_seed} -- see EXPECTED_OUTPUTS.md for how "
          f"much run-to-run variance to actually expect despite this.")

    branch = ask("Branch", default="chipseq", choices=["chipseq", "chrombpnet"])

    try:
        if branch == "chipseq":
            run_chipseq_interactive(cfg)
        else:
            run_chrombpnet_interactive(cfg)
    except FastaNotFoundError as e:
        # pyfaidx.FastaNotFoundError is an OSError, not a FileNotFoundError
        # -- a bare "except FileNotFoundError" (as cli.py used to have)
        # misses it entirely. cfg.genome_fasta is the one file the README
        # asks a fresh clone to download by hand, so this is the single
        # most likely first-run failure; catching it here at the top level
        # covers every Fasta(cfg.genome_fasta) call site at once.
        print(f"\n[STOP] genome FASTA not found or not indexable: {e}\n"
              f"(check cfg.genome_fasta -- currently {cfg.genome_fasta!r} -- points at a real, "
              f"readable hg38 FASTA; see README.md's Data setup section)")
        sys.exit(1)


if __name__ == "__main__":
    main()
