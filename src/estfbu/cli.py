"""Command-line entry point for both pipeline branches.

Usage:
    # has-ChIP-seq branch (design new sequences)
    python3 -m estfbu.cli --branch chipseq --tf GATA2 --cell-types HepG2,K562 --target specificity
    python3 -m estfbu.cli --branch chipseq --tf GATA2 --cell-types HepG2 --target activity

    # no-ChIP-seq / ChromBPNet branch (score + annotate existing regions)
    python3 -m estfbu.cli --branch chrombpnet --cell-types HepG2,K562 --target specificity --max-regions 5000
    python3 -m estfbu.cli --branch chrombpnet --cell-types HepG2 --target activity --max-regions 5000

    # no-ChIP-seq / ChromBPNet branch, design (TF-specific, no ChIP-seq needed)
    python3 -m estfbu.cli --branch chrombpnet --action design --tf GATA2 \\
        --cell-types HepG2,K562 --target specificity

    # no-ChIP-seq / ChromBPNet branch, genome-wide screen (no TF specified)
    python3 -m estfbu.cli --branch chrombpnet --action screen \\
        --cell-types HepG2,K562 --target specificity

    # any design run: also emit an esMPRA-shaped oligo library (see
    # oligo_library.py's docstring for the validation caveat)
    python3 -m estfbu.cli --branch chipseq --tf GATA2 --cell-types HepG2 --target activity \\
        --emit-oligo-library --oligo-pre ACTGGCCGCTTCACTG --oligo-after GGTACCTCTAGAGGATCCGG \\
        --barcode-pre CGTC --barcode-length 12
"""
import argparse
import sys
import time

import numpy as np
from pyfaidx import Fasta, FastaNotFoundError

from . import blacklist, chrombpnet_scoring, data_prep, design, motif_hits, postprocess, screen_tfs, train
from .config import load_config, seed_everything
from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks
from .oligo_library import DEFAULT_MAX_OLIGO_LEN, emit_oligo_library
from .reporting import write_bubble_plot, write_step_manifest


def _maybe_emit_oligo_library(args, cfg, out_fasta):
    """Shared by both branches' design output: if --emit-oligo-library was
    given, run the designed sequences through the esMPRA-shaped
    post-step (see oligo_library.py's docstring for the validation
    caveat) and report what it produced."""
    if not args.emit_oligo_library:
        return
    if not (args.oligo_pre and args.oligo_after and args.barcode_pre):
        print("[STOP] --emit-oligo-library needs --oligo-pre, --oligo-after, and --barcode-pre all "
              "given explicitly -- there's no safe default to assume silently here (a library with no "
              "adapters isn't a real esMPRA-shaped library, it just looks like one). See --oligo-pre's "
              "help for esMPRA's own reported adapter values if you don't have your own.")
        sys.exit(1)
    print(f"\n[oligo library] trimming/tiling {out_fasta} to fit esMPRA-style oligos "
          f"(max {args.max_oligo_len}bp incl. adapters + barcode)")
    inserts_out, oligos_out, manifest_out = emit_oligo_library(
        out_fasta, cfg.work_dir_path("results"),
        oligo_pre=args.oligo_pre, oligo_after=args.oligo_after,
        barcode_pre=args.barcode_pre, barcode_length=args.barcode_length,
        max_oligo_len=args.max_oligo_len, seed=cfg.training.random_seed,
        restriction_sites=cfg.postprocess.restriction_sites)
    print(f"  {inserts_out} (bare inserts, for esMPRA's --ref_fa)\n  {oligos_out} (array-order oligos)\n  {manifest_out}")


def run_chipseq(args, cfg):
    import pandas as pd

    t0 = time.time()
    cell_types = [c.strip() for c in args.cell_types.split(",") if c.strip()]

    if args.action == "screen":
        # the has-ChIP-seq branch's 'no TF specified' path: rank every
        # ChIP-seq-available, non-blacklisted TF by its own trained/
        # pretrained model's held-out score, instead of designing for one
        # given TF. Previously this logic (train.rank_chipseq_tfs) only
        # existed inline inside run_pipeline.py's interactive-only
        # run_chipseq_screen, wrapped around ask() calls -- unreachable
        # from the CLI and structurally untestable. This is the same
        # ranking function, called with args.target/args.n_top instead of
        # prompting for them.
        if args.target == "specificity" and len(cell_types) != 2:
            print("[STOP] --action screen --target specificity requires exactly 2 --cell-types")
            sys.exit(1)
        print(f"=== esTFBU [chipseq]: screen, cell_types={cell_types}, target={args.target} ===")
        valid_tfs = blacklist.exclude_blacklisted(blacklist.available_chipseq_tfs(cell_types, cfg), cfg)
        print(f"\n[1] {len(valid_tfs)} ChIP-seq-available, non-blacklisted TF(s): "
              f"{', '.join(valid_tfs) if valid_tfs else '(none)'}")
        if not valid_tfs:
            print("  [STOP] no TF has the data this run needs.")
            sys.exit(1)

        print(f"\n[2] Scoring {len(valid_tfs)} candidate TF(s) for target={args.target}")
        results = train.rank_chipseq_tfs(cfg, valid_tfs, cell_types, args.target)
        for r in results:
            if args.target == "specificity":
                print(f"  {r['tf']}: {cell_types[0]}={r['score_a']:.4f} auc={r['auc_a']:.3f} "
                      f"(tpm={r['tpm_a']:.1f}), {cell_types[1]}={r['score_b']:.4f} auc={r['auc_b']:.3f} "
                      f"(tpm={r['tpm_b']:.1f}), diff={r['score_a'] - r['score_b']:+.4f}")
            else:
                print(f"  {r['tf']}: {cell_types[0]}={r['score_a']:.4f} auc={r['auc_a']:.3f} (tpm={r['tpm_a']:.1f})")

        top = results[:args.n_top]
        print(f"\n  Top {args.n_top}: {[r['tf'] for r in top]}")
        out_csv = cfg.work_path("results", f"chipseq_screen_{args.target}_{'_'.join(cell_types)}.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        write_step_manifest(cfg, "screen", f"chipseq_{args.target}_{'_'.join(cell_types)}", out_csv,
                             "csv", len(results), cfg.training.random_seed, time.time() - t0)
        print(f"  {out_csv}")

        if args.target == "specificity":
            out_bubble = cfg.work_path("results", f"chipseq_screen_{args.target}_{'_'.join(cell_types)}_bubble.png")
            write_bubble_plot([r["tf"] for r in top], [r["score_a"] for r in top], [r["score_b"] for r in top],
                               cell_types, cfg, out_bubble,
                               f"Top {args.n_top} candidate TFs: bubble size = each TF's own trained-model\n"
                               f"score on its real ChIP-seq positives, color = RNA-seq expression")
            print(f"  {out_bubble}")

        print(f"\n=== Done: {out_csv} ===")
        return out_csv

    print(f"=== esTFBU [chipseq]: TF={args.tf}, cell_types={cell_types}, target={args.target} ===")

    print("\n[1] Blacklist check")
    try:
        blacklist.check(args.tf, cfg.blacklist_file)
    except blacklist.BlacklistedError as e:
        print(f"[BLOCKED] {e}")
        sys.exit(1)
    print(f"  ok: '{args.tf}' is not blacklisted")

    print("\n[2] Data availability + prep")
    for ct in cell_types:
        try:
            h5 = data_prep.prepare(args.tf, ct, cfg)
            print(f"  ok: {ct} data ready -> {h5}")
        except (FileNotFoundError, FastaNotFoundError) as e:
            print(f"  [STOP] {e}")
            sys.exit(1)

    print("\n[3] Genetic algorithm design")
    if args.starting_sequence:
        print(f"  seeding from given starting sequence ({len(args.starting_sequence)}bp)")
    seqs, scores = design.run_ga(args.tf, cell_types, args.target, cfg,
                                  starting_sequence=args.starting_sequence)
    print(f"  done: {len(seqs)} candidates in archive, best score={scores.max():.4f}")

    print("\n[4] Post-processing")
    final = postprocess.filter_and_dedup(args.tf, seqs, scores, cfg, target_gc=args.gc_target)
    print(f"  {len(final)} sequences survived filtering/dedup")

    out_fasta = cfg.work_path("results", f"{args.tf}_{'_'.join(cell_types)}_{args.target}.fasta")
    with open(out_fasta, "w") as f:
        for i, seq in enumerate(final):
            f.write(f">{args.tf}_{args.target}_{i}\n{seq}\n")
    write_step_manifest(cfg, "design", f"{args.tf}_{'_'.join(cell_types)}_{args.target}", out_fasta,
                         "fasta", len(final), cfg.training.random_seed, time.time() - t0)
    _maybe_emit_oligo_library(args, cfg, out_fasta)

    print(f"\n=== Done: {out_fasta} ===")
    return out_fasta


def run_chrombpnet(args, cfg):
    import pandas as pd

    t0 = time.time()
    cell_types = [c.strip() for c in args.cell_types.split(",") if c.strip()]
    print(f"=== esTFBU [chrombpnet]: cell_types={cell_types}, target={args.target}, "
          f"max_regions={args.max_regions} ===")

    if args.tf:
        print("\n[1] Blacklist check")
        try:
            blacklist.check(args.tf, cfg.blacklist_file)
        except blacklist.BlacklistedError as e:
            print(f"[BLOCKED] {e}")
            sys.exit(1)
        print(f"  ok: '{args.tf}' is not blacklisted")
    else:
        print("\n[1] Blacklist check: skipped (no --tf given, this is a general region scan)")

    if args.action == "screen":
        print(f"\n[2] Genome-wide TF screening (no TF specified -- ranking all 198 known TFs)")
        # screen_all_tfs copies the bundled precomputed scan into place
        # itself if it covers this cell-type pair (column-checked) --
        # this used to be checked here without ever reading
        # cfg.precomputed_genome_scan, so the CLI alone could never reach
        # the bundled scan the way the interactive script could.
        scan_csv = cfg.work_dir_path("results") / f"chrombpnet_full_genome_scan_{cell_types[0].lower()}_atac_peaks.csv"
        try:
            result_df = screen_tfs.screen_all_tfs(cfg, cell_types, scan_csv_path=scan_csv, mode=args.target)
        except ValueError as e:
            print(f"  [STOP] {e}\n  (needs a full genome-wide ChromBPNet scan for {cell_types} to "
                  f"already exist -- see test_scripts/chrombpnet_full_genome_scan.py <cell_type_a> "
                  f"<cell_type_b> -- this takes several hours, so it's not triggered automatically here).")
            sys.exit(1)
        out_csv = cfg.work_path("results", f"tf_screen_{args.target}_{'_'.join(cell_types)}.csv")
        result_df.to_csv(out_csv, index=False)
        n_sig = (result_df["pvalue"] < 0.05).sum()
        print(f"  screened {len(result_df)} TFs, {n_sig} with uncorrected p<0.05 (no multiple-testing "
              f"correction applied across 198 tests -- read this as a rough filter, not a real FDR)")
        write_step_manifest(cfg, "screen", f"chrombpnet_{args.target}_{'_'.join(cell_types)}", out_csv,
                             "csv", len(result_df), cfg.training.random_seed, time.time() - t0)

        if args.target == "specificity" and len(cell_types) == 2:
            # ranked by signed effect size now (see screen_all_tfs), so
            # the top rows are already the strongest cell_types[0]-favoring
            # hits -- same bubble plot the interactive script produces,
            # previously unreachable from the CLI since write_bubble_plot
            # only existed in run_pipeline.py
            print(f"\n[3] Bubble plot: top {args.n_top} candidates")
            top_tfs = result_df[result_df["pvalue"].notna()].head(args.n_top)["tf"].tolist()
            genome = Fasta(cfg.genome_fasta)
            scan_df = pd.read_csv(scan_csv)
            chroms, centers = scan_df["chrom"].to_numpy(), scan_df["center"].to_numpy()
            scores_a, scores_b = [], []
            for tf in top_tfs:
                motif_scores = screen_tfs._score_tf_genome_wide(tf, genome, chroms, centers,
                                                                  cfg.jaspar_pfm_cache, cfg.model.seq_len)
                valid = ~np.isnan(motif_scores)
                threshold = np.nanquantile(motif_scores, 0.9)
                strong = valid & (motif_scores >= threshold)
                scores_a.append(scan_df[f"{cell_types[0].lower()}_score"][strong].mean())
                scores_b.append(scan_df[f"{cell_types[1].lower()}_score"][strong].mean())
            out_bubble = cfg.work_path("results", f"tf_screen_{args.target}_{'_'.join(cell_types)}_bubble.png")
            write_bubble_plot(top_tfs, scores_a, scores_b, cell_types, cfg, out_bubble,
                               f"Top {args.n_top} candidate TFs: bubble size = ChromBPNet accessibility\n"
                               f"in that TF's strong-motif regions, color = RNA-seq expression")
            print(f"  {out_bubble}")
        print(f"\n=== Done: {out_csv} ===")
        return out_csv

    if args.action == "design":
        if args.tf:
            print(f"\n[2] Genetic algorithm design (fitness = gradient-based motif-importance "
                  f"for '{args.tf}', the ChIP-seq-free proxy -- see design.run_ga_chrombpnet's docstring)")
        else:
            print(f"\n[2] Genetic algorithm design (ChromBPNet-driven, no TF-specific masking -- "
                  f"pass --tf to switch to TF-specific fitness)")
        if args.starting_sequence:
            print(f"  seeding from given starting sequence ({len(args.starting_sequence)}bp)")
            if args.auto_fix_top_fraction:
                print(f"  auto-protecting top {args.auto_fix_top_fraction:.0%} of positions by "
                      f"gradient importance, unioned with any lowercase-marked positions")
        seqs, scores = design.run_ga_chrombpnet(cell_types, args.target, cfg,
                                                 starting_sequence=args.starting_sequence,
                                                 tf=args.tf, auto_fix_top_fraction=args.auto_fix_top_fraction)
        print(f"  done: {len(seqs)} candidates in archive, best score={scores.max():.4f}")

        print(f"\n[3] Post-processing (restriction sites + edit-distance dedup -- no TF-motif "
              f"checks, ChromBPNet design has no TF-specific core motif to check against)")
        final = postprocess.dedup_by_edit_distance(seqs, scores, cfg, check_restriction_sites=True)
        print(f"  {len(final)} sequences survived filtering/dedup")

        name_prefix = f"chrombpnet_design_{args.tf}_{'_'.join(cell_types)}" if args.tf else \
                      f"chrombpnet_design_{'_'.join(cell_types)}"
        out_fasta = cfg.work_path("results", f"{name_prefix}_{args.target}.fasta")
        with open(out_fasta, "w") as f:
            for i, seq in enumerate(final):
                f.write(f">chrombpnet_{args.target}_{i}\n{seq}\n")
        write_step_manifest(cfg, "design", f"{name_prefix}_{args.target}", out_fasta,
                             "fasta", len(final), cfg.training.random_seed, time.time() - t0)
        _maybe_emit_oligo_library(args, cfg, out_fasta)
        print(f"\n=== Done: {out_fasta} ({len(final)} sequences, deduplicated and filtered) ===")
        return out_fasta

    print(f"\n[2] Scanning {args.max_regions} {cell_types[0]} ATAC peaks with the {cell_types[0]} ChromBPNet model")
    regions, primary_scores = chrombpnet_scoring.scan_atac_peaks(cell_types[0], cfg, max_regions=args.max_regions)
    print(f"  scored {len(regions)} regions")

    df = pd.DataFrame({
        "chrom": [r[0] for r in regions],
        "center": [r[1] for r in regions],
        f"{cell_types[0]}_score": primary_scores,
    })

    if args.target == "specificity":
        if len(cell_types) != 2:
            print("\n[STOP] --target specificity requires exactly 2 --cell-types")
            sys.exit(1)
        print(f"\n[3] Scoring the same regions with the {cell_types[1]} model")
        secondary_scores = chrombpnet_scoring.score_regions_bulk(regions, cell_types[1], cfg)
        df[f"{cell_types[1]}_score"] = secondary_scores
        df["diff"] = df[f"{cell_types[0]}_score"] - df[f"{cell_types[1]}_score"]
        df = df.dropna().sort_values("diff", ascending=False)
    else:
        print("\n[3] target=activity: skipping second cell type comparison")
        df = df.dropna().sort_values(f"{cell_types[0]}_score", ascending=False)

    if args.tf:
        print(f"\n[3.5] Restricting to '{args.tf}': re-ranking regions by its own motif match")
        ppm = trim_low_information_flanks(load_ppm(args.tf, cfg.jaspar_pfm_cache))
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
        df[f"{args.tf}_motif_score"] = motif_scores
        df = df.dropna(subset=[f"{args.tf}_motif_score"]).sort_values(f"{args.tf}_motif_score", ascending=False)
        print(f"  {len(df)} regions re-ranked by {args.tf} motif match strength")

    print(f"\n[4] Annotating top/bottom {args.annotate_top} regions with candidate TF matches "
          f"(slow step: {args.annotate_top * 2} regions x 198 TF motifs x ~100 background shuffles each)")
    print("  ranked by empirical p-value against a composition-matched shuffled-sequence "
          "background, not raw match score alone -- some short/low-information JASPAR motifs "
          "(e.g. SP5, MEIS1) match broadly by chance and would otherwise dominate every region")
    subset = pd.concat([df.head(args.annotate_top), df.tail(args.annotate_top)])
    annotated = motif_hits.annotate_regions(subset, cfg, top_n=3, cell_types=cell_types)

    out_csv = cfg.work_path("results", f"chrombpnet_{'_'.join(cell_types)}_{args.target}.csv")
    df.to_csv(out_csv, index=False)
    out_annotated_csv = cfg.work_path("results", f"chrombpnet_{'_'.join(cell_types)}_{args.target}_annotated_top{args.annotate_top}.csv")
    annotated.to_csv(out_annotated_csv, index=False)
    write_step_manifest(cfg, "score", f"chrombpnet_{'_'.join(cell_types)}_{args.target}", out_csv,
                         "csv", len(df), cfg.training.random_seed, time.time() - t0)

    print(f"\n=== Done: {out_csv} (full scan), {out_annotated_csv} (top/bottom annotated) ===")
    return out_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one branch of the esTFBU pipeline.")
    ap.add_argument("--branch", choices=["chipseq", "chrombpnet"], default="chipseq")
    ap.add_argument("--tf", default=None, help="Required for --branch chipseq; optional for chrombpnet")
    ap.add_argument("--cell-types", required=True, help="Comma-separated, e.g. HepG2 or HepG2,K562")
    ap.add_argument("--target", choices=["activity", "specificity"], default="specificity")
    ap.add_argument("--config", default=None, help="Path to a config YAML (default: config/default_config.yaml)")
    ap.add_argument("--gc-target", type=float, default=0.5,
                     help="[chipseq] Target GC content for filtering")
    ap.add_argument("--starting-sequence", default=None,
                     help="Optional exact-length starting sequence to seed/anchor the GA search "
                          "around instead of designing de novo. Required length: "
                          "config.model.seq_len (168bp default) for --branch chipseq, or exactly "
                          "2114bp for --branch chrombpnet --action design. Lowercase letters mark "
                          "positions to keep fixed for the whole run; uppercase stays free to optimize.")
    ap.add_argument("--auto-fix-top-fraction", type=float, default=None,
                     help="[chrombpnet --action design, needs --starting-sequence] Also auto-detect "
                          "and protect the top fraction of positions by gradient importance (e.g. "
                          "0.05 for the top 5%%), unioned with any lowercase-marked positions -- "
                          "not a replacement for manual marking, opt-in on top of it")
    ap.add_argument("--max-regions", type=int, default=5000,
                     help="[chrombpnet] Number of ATAC peaks to scan (full genome-wide coverage "
                          "is ~170k peaks/cell-type, ~4.5hr each at measured rate)")
    ap.add_argument("--emit-oligo-library", action="store_true",
                     help="After any design run, also trim/tile the designed sequences to fit an "
                          "esMPRA-style array-synthesis oligo (adapters + barcode), spike in "
                          "scrambled-sequence negative controls, and write that FASTA + a manifest "
                          "alongside the raw design output -- see oligo_library.py's docstring for "
                          "the validation caveat (implemented from esMPRA's documented interface, "
                          "not tested against esMPRA's own source in this environment)")
    ap.add_argument("--oligo-pre", default="",
                     help="[--emit-oligo-library] 5' adapter/primer to prepend. No default is set here "
                          "because it isn't independently verified in this environment (no esMPRA "
                          "source available), but a review pass reported esMPRA's own published "
                          "adapters as GGCCGCTTGACG (oligo_pre) / CACTGCGGCTCC (oligo_after) / "
                          "CGAACCTCTAGA (barcode_pre) -- worth trying if you don't have your own.")
    ap.add_argument("--oligo-after", default="", help="[--emit-oligo-library] 3' adapter/primer to append")
    ap.add_argument("--barcode-pre", default="", help="[--emit-oligo-library] fixed spacer right before the random barcode")
    ap.add_argument("--barcode-length", type=int, default=12,
                     help="[--emit-oligo-library] random barcode length. A review pass reported "
                          "esMPRA's own barcode length as 20 (unverified in this environment).")
    ap.add_argument("--max-oligo-len", type=int, default=DEFAULT_MAX_OLIGO_LEN,
                     help="[--emit-oligo-library] total oligo length budget including adapters + barcode "
                          "(array synthesis typically tops out ~230-300bp)")
    ap.add_argument("--action", choices=["score", "design", "screen"], default="score",
                     help="[chrombpnet] 'score' scans+annotates existing ATAC peaks (default); "
                          "'design' runs a genetic algorithm to generate new sequences optimized "
                          "for predicted accessibility (activity) or differential accessibility "
                          "(specificity), reusing --starting-sequence if given; 'screen' ranks all "
                          "198 known TFs genome-wide by motif enrichment in differential "
                          "(--target specificity) or high-accessibility (--target activity) "
                          "regions, with NO --tf needed -- requires a full genome-wide ChromBPNet "
                          "scan to already exist (see test_scripts/chrombpnet_full_genome_scan.py "
                          "<cell_type_a> <cell_type_b>). "
                          "[chipseq] 'design' (default effective behavior) runs the GA for --tf; "
                          "'screen' ranks every ChIP-seq-available, non-blacklisted TF by its own "
                          "trained/pretrained model's held-out-test-split score, with NO --tf "
                          "needed -- this is the has-ChIP-seq branch's 'no TF specified' path, "
                          "previously only reachable from the interactive run_pipeline.py.")
    ap.add_argument("--annotate-top", type=int, default=10,
                     help="[chrombpnet] How many top+bottom differential regions to run TF annotation on")
    ap.add_argument("--n-top", type=int, default=3,
                     help="[chipseq/chrombpnet --action screen, target=specificity, 2 cell types] How "
                          "many top-ranked candidates to keep / render in the bubble plot ('best 1' or "
                          "'best 3' style)")
    args = ap.parse_args(argv)

    if args.emit_oligo_library and not (args.oligo_pre and args.oligo_after and args.barcode_pre):
        # fail fast, before any real GA/data-prep work runs -- this used
        # to only be checked in _maybe_emit_oligo_library, called after a
        # full design run completes, so a missing adapter wasted the
        # entire run (minutes to tens of minutes) before saying so.
        print("[STOP] --emit-oligo-library needs --oligo-pre, --oligo-after, and --barcode-pre all "
              "given explicitly -- there's no safe default to assume silently here (a library with no "
              "adapters isn't a real esMPRA-shaped library, it just looks like one). See --oligo-pre's "
              "help for esMPRA's own reported adapter values if you don't have your own.")
        sys.exit(1)

    cfg = load_config(args.config)
    seed_everything(cfg.training.random_seed)

    try:
        if args.branch == "chipseq":
            if not args.tf and args.action != "screen":
                print("[STOP] --branch chipseq requires --tf (unless --action screen)")
                sys.exit(1)
            return run_chipseq(args, cfg)
        else:
            return run_chrombpnet(args, cfg)
    except FastaNotFoundError as e:
        # pyfaidx.FastaNotFoundError is an OSError, NOT a FileNotFoundError
        # (verified: its MRO is [FastaNotFoundError, OSError, Exception]) --
        # so it slips past a bare "except FileNotFoundError" undetected.
        # cfg.genome_fasta is read from ~10 call sites across both branches
        # (data prep, GA seeding, motif scanning, screening, bubble plots),
        # and it's the one file the README asks a fresh clone to download
        # by hand -- so a missing/misconfigured genome_fasta was, in
        # practice, the single most likely first-run failure, and it was
        # escaping every [STOP]-message guard in this file. Catching it
        # here at the top level covers every call site at once instead of
        # rewrapping each one individually.
        print(f"[STOP] genome FASTA not found or not indexable: {e}\n"
              f"(check cfg.genome_fasta -- currently {cfg.genome_fasta!r} -- points at a real, "
              f"readable hg38 FASTA; see README.md's Data setup section)")
        sys.exit(1)


if __name__ == "__main__":
    main()
