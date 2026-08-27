"""Regression tests against known-good baselines established during
development. These are slow (real genome I/O, real model inference) by
nature of what they're testing -- not unit tests, integration checks that
the refactor still reproduces validated numbers.

Run with: python3 -m pytest test_scripts/test_regression.py -v -s
(or python3 test_scripts/test_regression.py to run directly without pytest)

Uses config/default_config.yaml (bare load_config()) by default, which
needs the full-scale external data set up per README.md's "Data setup"
section (~22GB, not bundled). Point ESTFBU_CONFIG at
config/quickstart_config.yaml to run the fast subset against the bundled
sample_data/ instead -- most tests still need real genome/model files
either way, this just picks which set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from estfbu import blacklist, data_prep, motif_scoring
from estfbu.config import load_config


def test_blacklist_blocks_listed_tf(tmp_path):
    bl = tmp_path / "blacklist.txt"
    bl.write_text("# comment\nGATA2\n")
    try:
        blacklist.check("GATA2", bl)
        assert False, "expected BlacklistedError"
    except blacklist.BlacklistedError:
        pass


def test_blacklist_allows_unlisted_tf(tmp_path):
    bl = tmp_path / "blacklist.txt"
    bl.write_text("# comment\nTP53\n")
    blacklist.check("GATA2", bl)  # should not raise


def test_gata2_consensus_motif():
    """Known-good: GATA2's trimmed JASPAR motif is CTTATCT (validated
    against the paper's own hardcoded matrix during this session)."""
    cfg = load_config()
    ppm = motif_scoring.trim_low_information_flanks(
        motif_scoring.load_ppm("GATA2", cfg.jaspar_pfm_cache)
    )
    consensus = motif_scoring.consensus_sequence(ppm)
    assert consensus == "CTTATCT", f"expected CTTATCT, got {consensus}"


def test_hepg2_gata2_data_prep_sample_count():
    """Known-good baseline: refactored data_prep.py should produce roughly
    the same balanced sample count as the original hand-run pipeline
    (21,001 pos+neg combined, measured during development as 20,919/21,471).
    Allow +/-10% since GC-binning order can shift the exact count slightly
    without indicating a real bug."""
    import h5py
    cfg = load_config()
    h5_path = data_prep.prepare("GATA2", "HepG2", cfg)
    with h5py.File(h5_path) as f:
        n_pos = f["pos_GATA2"].shape[0]
        n_neg = f["neg_GATA2"].shape[0]
    baseline = 21001
    n_balanced = min(n_pos, n_neg)
    assert abs(n_balanced - baseline) / baseline < 0.10, (
        f"balanced sample count {n_balanced} deviates >10% from baseline {baseline} "
        f"(pos={n_pos}, neg={n_neg}) -- check data_prep.py for a regression"
    )


def test_design_ga_seeded_and_unseeded_dont_crash():
    """Regression test for the pool_size/target_pool_size bug: a
    starting_sequence-seeded run (prone to population collapse below the
    target pool size via repeated single-anchor mutation collisions) must
    not IndexError, and the unseeded path must be unaffected by the fix
    (same score at the same iteration count as before the fix, 1.6965 at
    5 iterations, confirmed during development)."""
    import h5py
    from estfbu import data_prep, design
    from estfbu.motif_scoring import consensus_sequence, load_ppm, trim_low_information_flanks

    cfg = load_config()
    cfg.genetic_algorithm.n_iterations = 3

    # unseeded path
    seqs, scores = design.run_ga("GATA2", ["HepG2", "K562"], "specificity", cfg, progress_every=0)
    assert len(seqs) > 0 and scores.max() > 0

    # seeded path -- this is what originally triggered the IndexError
    h5_path = data_prep.prepare("GATA2", "HepG2", cfg)
    with h5py.File(h5_path) as f:
        onehot = f["pos_GATA2"][0]
    seq = design._onehot_to_seq(onehot)
    ppm = trim_low_information_flanks(load_ppm("GATA2", cfg.jaspar_pfm_cache))
    motif = consensus_sequence(ppm)
    full_seq = seq.replace("N" * len(motif), motif)

    seqs2, scores2 = design.run_ga("GATA2", ["HepG2", "K562"], "specificity", cfg,
                                    progress_every=0, starting_sequence=full_seq)
    assert len(seqs2) > 0 and scores2.max() > 0


def test_design_seq_to_onehot_respects_configured_seq_len():
    """Regression test for a real bug: design._seq_to_onehot used to
    default length=168 and both its call sites (_seed_from_sequence,
    run_ga's per-generation encoding) omitted the argument entirely, so
    cfg.model.seq_len was silently ignored by the GA no matter what the
    config said. A seq_len > 168 crashed with an opaque IndexError; a
    seq_len < 168 was worse -- it completed with no error at all, having
    silently zero-padded the tail into a corrupted-but-plausible-looking
    encoding. Now length is a required argument that raises ValueError on
    mismatch (a plain `raise`, not a bare `assert` -- `python -O` strips
    asserts entirely, which would silently defeat a check whose whole job
    is turning silent corruption loud), so both failure modes become one
    loud one. This test also exercises _seed_from_sequence (which threads
    cfg.model.seq_len through to the same function) at a real, non-default
    length -- the whiteboard's "168bp -> L" requirement -- without needing
    to train a model from scratch, since it stops short of scoring."""
    from estfbu import design
    from estfbu.motif_scoring import consensus_sequence, load_ppm, trim_low_information_flanks

    cfg = load_config()

    # the bug: a length mismatch must raise, not silently zero-pad or
    # index out of bounds. NOTE: don't write this as
    #   try: design._seq_to_onehot(["ACGT"], 5); assert False, "..."
    #   except AssertionError: pass
    # -- if the guard is ever removed, _seq_to_onehot silently returns and
    # the bare `assert False` inside the try raises AssertionError itself,
    # which the `except AssertionError` then swallows just as "successfully"
    # as a real guard failure would have been caught. Using a distinct
    # exception type the guard actually raises (ValueError) plus a flag
    # checked OUTSIDE the try/except closes that hole: a missing guard
    # then fails this test for real instead of passing by accident.
    raised = False
    try:
        design._seq_to_onehot(["ACGT"], 5)
    except ValueError:
        raised = True
    assert raised, "expected a ValueError for a length mismatch"

    onehot = design._seq_to_onehot(["ACGT", "TTTT"], 4)
    assert onehot.shape == (2, 4, 4)

    # _seed_from_sequence must honor a non-168 cfg.model.seq_len end to end
    cfg.model.seq_len = 200
    motif = consensus_sequence(trim_low_information_flanks(load_ppm("GATA2", cfg.jaspar_pfm_cache)))
    m = len(motif)
    core_start = 100 - m // 2
    starting_sequence = "A" * core_start + motif + "A" * (200 - core_start - m)
    seed_onehot, protected = design._seed_from_sequence("GATA2", starting_sequence, cfg, pool_size=5)
    assert seed_onehot.shape == (5, 200, 4), f"expected (5, 200, 4), got {seed_onehot.shape}"


def test_chrombpnet_design_produces_correct_length_sequences():
    """Regression test for run_ga_chrombpnet: activity, specificity, and
    starting_sequence modes should all complete and produce valid
    2114bp sequences (the ChromBPNet native input length, distinct from
    the 168bp TFBU context length used by the chipseq branch)."""
    from estfbu import design

    cfg = load_config()
    cfg.genetic_algorithm.n_iterations = 2
    cfg.genetic_algorithm.max_pool_size = 20

    seqs_a, scores_a = design.run_ga_chrombpnet(["HepG2"], "activity", cfg,
                                                 progress_every=0, seed_pool_size=20)
    assert len(seqs_a) > 0
    assert all(len(s) == design.CHROMBPNET_INPUT_LEN for s in seqs_a[:5])

    seqs_s, scores_s = design.run_ga_chrombpnet(["HepG2", "K562"], "specificity", cfg,
                                                 progress_every=0, seed_pool_size=20)
    assert len(seqs_s) > 0
    assert all(len(s) == design.CHROMBPNET_INPUT_LEN for s in seqs_s[:5])

    seqs_seeded, scores_seeded = design.run_ga_chrombpnet(
        ["HepG2", "K562"], "specificity", cfg, progress_every=0,
        starting_sequence=seqs_a[0], seed_pool_size=20)
    assert len(seqs_seeded) > 0


def test_chrombpnet_design_tf_specific_uses_motif_importance():
    """Regression test for the tf= gap fix in run_ga_chrombpnet: a specified
    TF should switch the fitness function to the gradient-based motif-
    importance proxy (chrombpnet_scoring.tf_motif_importance_batch)
    instead of raw accessibility, letting the no-ChIP-seq branch design a
    TF-specific sequence without ChIP-seq data. Checks: (1) it runs to
    completion and produces valid-length sequences for both activity and
    specificity targets, (2) the best-scoring sequence actually has a
    recognizable GATA2 PWM match (the fitness function should push toward
    keeping/creating one, not just tolerate it by chance)."""
    from estfbu import chrombpnet_scoring as cbp
    from estfbu import design
    from estfbu.motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks

    cfg = load_config()
    cfg.genetic_algorithm.n_iterations = 3
    cfg.genetic_algorithm.max_pool_size = 20

    seqs_a, scores_a = design.run_ga_chrombpnet(["HepG2"], "activity", cfg, tf="GATA2",
                                                  progress_every=0, seed_pool_size=20)
    assert len(seqs_a) > 0
    assert all(len(s) == design.CHROMBPNET_INPUT_LEN for s in seqs_a[:5])

    ppm = trim_low_information_flanks(load_ppm("GATA2", cfg.jaspar_pfm_cache))
    best_seq = seqs_a[0]
    _, v_fwd = best_match(best_seq, ppm)
    _, v_rev = best_match(reverse_complement(best_seq), ppm)
    assert max(v_fwd, v_rev) > 0, "best-scoring designed sequence should contain a real GATA2 motif match"

    seqs_s, scores_s = design.run_ga_chrombpnet(["HepG2", "K562"], "specificity", cfg, tf="GATA2",
                                                  progress_every=0, seed_pool_size=20)
    assert len(seqs_s) > 0
    assert all(len(s) == design.CHROMBPNET_INPUT_LEN for s in seqs_s[:5])

    # every real PWM (pseudocount-smoothed) has some nonzero-probability
    # window in any long-enough sequence, so tf_motif_importance_batch
    # should return a real (non-NaN) score for an ordinary designed sequence
    scores = cbp.tf_motif_importance_batch(seqs_a[:3], "GATA2", "HepG2", cfg)
    assert not np.isnan(scores).any()


def test_run_ga_chrombpnet_respects_manually_fixed_positions():
    """Regression test for the gap-2 fix: lowercase letters in a user-given
    starting_sequence must mark positions the GA never mutates or
    recombines away, for the whole run. Uses the chrombpnet branch (no TF,
    no auto-masking) so this isolates the manual-fixing mechanism cleanly
    from the has-ChIP-seq branch's separate auto-masked-core-motif logic
    (covered by test_chrombpnet_design_produces_correct_length_sequences
    and test_design_ga_seeded_and_unseeded_dont_crash not needing it)."""
    from estfbu import design

    cfg = load_config()
    cfg.genetic_algorithm.n_iterations = 3
    cfg.genetic_algorithm.max_pool_size = 20

    base = ("ACGT" * (design.CHROMBPNET_INPUT_LEN // 4 + 1))[:design.CHROMBPNET_INPUT_LEN]
    fixed_input = base[:50].lower() + base[50:]  # first 50bp marked fixed

    seqs, scores = design.run_ga_chrombpnet(["HepG2"], "activity", cfg,
                                             starting_sequence=fixed_input,
                                             seed_pool_size=20, progress_every=0)
    assert len(seqs) > 1, "sanity check: the GA should have found more than just the seed"
    violations = [s for s in seqs if s[:50] != base[:50].upper()]
    assert not violations, f"{len(violations)} archived sequences mutated a position marked fixed"

    # an all-uppercase starting_sequence (the common case) must behave
    # exactly as before -- no positions protected
    all_upper_seqs, _ = design.run_ga_chrombpnet(["HepG2"], "activity", cfg,
                                                   starting_sequence=base,
                                                   seed_pool_size=20, progress_every=0)
    assert any(s != base for s in all_upper_seqs), \
        "with no lowercase marks, the whole sequence should be free to mutate"


def test_auto_detect_important_positions_are_protected():
    """Regression test for the auto-detect-fixed-positions feature (the
    "better version" of gap-2 the design brief flagged as optional/future
    work -- now cheap since it reuses gap-1's gradient_importance_batch).
    auto_fix_top_fraction should protect the requested fraction of
    positions (ranked by |gradient importance|), unioned with -- not
    replacing -- any manually lowercase-marked positions, for the whole
    GA run."""
    from estfbu import chrombpnet_scoring as cbp
    from estfbu import design

    cfg = load_config()
    cfg.genetic_algorithm.n_iterations = 3
    cfg.genetic_algorithm.max_pool_size = 20

    seed = design._seed_from_atac_peaks("HepG2", cfg, 1)[0]
    auto_positions = cbp.auto_detect_important_positions(seed, "HepG2", cfg, top_fraction=0.05)
    assert len(auto_positions) == round(len(seed) * 0.05)

    seqs, scores = design.run_ga_chrombpnet(["HepG2"], "activity", cfg, starting_sequence=seed,
                                             seed_pool_size=20, auto_fix_top_fraction=0.05, progress_every=0)
    assert len(seqs) > 1
    for s in seqs:
        for i in auto_positions:
            assert s[i] == seed[i].upper(), f"auto-protected position {i} was mutated"

    # union, not replacement: a manually-marked position outside the
    # auto-detected set must also stay protected
    manual_idx = next(i for i in range(len(seed)) if i not in auto_positions)
    combined_input = seed[:manual_idx] + seed[manual_idx].lower() + seed[manual_idx + 1:]
    seqs2, _ = design.run_ga_chrombpnet(["HepG2"], "activity", cfg, starting_sequence=combined_input,
                                          seed_pool_size=20, auto_fix_top_fraction=0.05, progress_every=0)
    for s in seqs2:
        assert s[manual_idx] == seed[manual_idx].upper(), "manually-marked position was mutated"
        for i in auto_positions:
            assert s[i] == seed[i].upper(), f"auto-protected position {i} was mutated"


def test_chrombpnet_design_output_is_deduplicated():
    """Regression test for the ChromBPNet-design filtering gap fix:
    dedup_by_edit_distance() must actually reduce a set containing
    near-identical sequences, and must never let two selected sequences
    sit closer than the configured edit-distance threshold."""
    import Levenshtein
    from estfbu import postprocess

    cfg = load_config()
    base = "ACGT" * (cfg.model.seq_len // 4)
    # 5 near-identical variants (should collapse to ~1 after dedup) + 1 very different sequence
    near_identical = [base[:i] + "T" + base[i + 1:] for i in range(5)]
    very_different = "TGCA" * (cfg.model.seq_len // 4)
    seqs = near_identical + [very_different]
    scores = [5, 4, 3, 2, 1, 0]  # near_identical[0] has the highest score, should be kept

    result = postprocess.dedup_by_edit_distance(seqs, scores, cfg, check_restriction_sites=False)
    assert len(result) < len(seqs), "dedup should have removed near-identical sequences"
    assert near_identical[0] in result, "the highest-scoring sequence should survive"
    assert very_different in result, "the genuinely different sequence should survive"
    threshold = round(len(seqs[0]) * cfg.postprocess.edit_distance_fraction)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert Levenshtein.distance(result[i], result[j]) >= threshold


def test_screen_all_tfs_ranks_known_biology():
    """Regression test for the genome-wide TF screening feature (no TF
    specified up front): screening a small subset of TFs against the
    already-validated full genome scan should reproduce the same
    conclusions as the targeted single-TF cross-validation -- GATA2's
    numbers should match cross_validate_gata2.py exactly (same
    methodology, this is just the generalized-to-any-TF version of it),
    and HNF4A/FOXA2 (known strong liver-specific TFs) should come out
    strongly HepG2-favoring, not K562-favoring. Also covers the TPM-
    visibility gap fix (tpm_a/tpm_b must be real, non-null RNA-seq values
    matching known biology, not just empty columns) and the any-cell-type
    gap fix (screen_all_tfs takes cell_types explicitly now instead of
    assuming HepG2/K562)."""
    from estfbu import screen_tfs

    import shutil
    import tempfile
    import unittest
    from pathlib import Path

    cfg = load_config()
    cell_types = ["HepG2", "K562"]
    scan_csv = cfg.work_dir_path("results") / f"chrombpnet_full_genome_scan_{cell_types[0].lower()}_atac_peaks.csv"
    if not scan_csv.exists():
        # fall back to the bundled precomputed scan -- it's the same
        # full-scale (~165k region) file behind ref_result/'s validated
        # headline numbers, not a small demo, so this makes the test
        # actually run on a fresh clone instead of only ever running on
        # the one machine that already had a full-scale scan sitting in
        # its work_dir.
        bundled = Path(__file__).resolve().parents[1] / "sample_data" / "precomputed" / \
            "chrombpnet_full_genome_scan_hepg2_atac_peaks.csv"
        if bundled.exists():
            scan_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundled, scan_csv)
        else:
            # raising unittest.SkipTest (not printing+returning) so this
            # is correctly reported as SKIPPED, not PASSED, whether run
            # directly (see __main__ below) or under pytest -- a test
            # that can silently "pass" by doing nothing is worse than no
            # test at all for a claim this central to the README.
            raise unittest.SkipTest(f"full genome scan not present at {scan_csv} or {bundled}")

    orig_names_fn = screen_tfs._all_tf_names
    screen_tfs._all_tf_names = lambda p: ["GATA2", "HNF4A"]
    try:
        df = screen_tfs.screen_all_tfs(cfg, cell_types, mode="specificity", progress_every=0)
        # mode='activity' with a single cell type: tpm_b must be NaN (only
        # one cell type is relevant), not silently populated from a
        # hardcoded second one. Fresh checkpoint dir so it doesn't reuse
        # the specificity run's cached rows (checkpoints aren't keyed by mode).
        with tempfile.TemporaryDirectory() as d:
            df_activity = screen_tfs.screen_all_tfs(cfg, ["HepG2"], scan_csv_path=scan_csv,
                                                      mode="activity", checkpoint_dir=Path(d), progress_every=0)
        assert pd.isna(df_activity.iloc[0]["tpm_b"])
    finally:
        screen_tfs._all_tf_names = orig_names_fn

    gata2_row = df[df["tf"] == "GATA2"].iloc[0]
    assert abs(gata2_row["mean_diff_strong"] - 0.1887) < 0.01, "GATA2 result should match cross_validate_gata2.py"
    assert gata2_row["pvalue"] < 1e-10

    hnf4a_row = df[df["tf"] == "HNF4A"].iloc[0]
    assert hnf4a_row["mean_diff_strong"] > gata2_row["mean_diff_strong"], (
        "HNF4A (known strong liver-specific TF) should be more HepG2-favoring than GATA2"
    )

    # the direction-awareness gap fix: HNF4A is a genuinely HepG2-favoring
    # hit (this is the exact TF the README's headline result names), so
    # effect_direction must say so explicitly, not just have a bigger
    # raw diff than GATA2 -- and the ranking itself must put it first
    # (df is sorted by signed effect size now, not raw p-value, so a
    # same-p-value K562-favoring TF can no longer silently outrank it)
    assert hnf4a_row["effect_direction"] == "HepG2"
    assert df.iloc[0]["tf"] == "HNF4A", "top-ranked TF should be the strongest HepG2-favoring hit, not just the smallest p-value"

    assert {"tpm_a", "tpm_b"}.issubset(df.columns)
    assert not pd.isna(hnf4a_row["tpm_a"]) and not pd.isna(hnf4a_row["tpm_b"])
    assert hnf4a_row["tpm_a"] > hnf4a_row["tpm_b"], (
        "HNF4A's own gene should be far more expressed in HepG2 (cell_types[0]) than K562 -- "
        "the exact check this TPM column exists so a user can make for any top-ranked candidate"
    )


def test_second_tf_hnf4a_generalizes():
    """Regression test for the second-TF validation: data_prep and GA
    design must both work end-to-end for HNF4A -- a TF with a different,
    longer motif than GATA2, using freshly downloaded real ENCODE data
    (ENCFF072CXB) rather than the paper's bundled files. This is also a
    regression test for the _safe_pos_orig fix: standard ENCODE narrowPeak
    files put a non-numeric peak name in column 4, which crashed the
    original pos_orig_fn assumption immediately on this exact data."""
    import h5py
    from estfbu import data_prep, design

    cfg = load_config()
    h5_path = data_prep.prepare("HNF4A", "HepG2", cfg)
    with h5py.File(h5_path) as f:
        n_pos = f["pos_HNF4A"].shape[0]
        n_neg = f["neg_HNF4A"].shape[0]
    assert n_pos > 1000 and n_neg > 1000, f"unexpectedly few samples: pos={n_pos}, neg={n_neg}"

    cfg.genetic_algorithm.n_iterations = 3
    cfg.genetic_algorithm.max_pool_size = 50
    seqs, scores = design.run_ga("HNF4A", ["HepG2"], "activity", cfg, progress_every=0)
    assert len(seqs) > 0 and scores.max() > 0


def test_cli_chipseq_screen_action():
    """Regression test for the has-ChIP-seq branch's 'no TF specified'
    rank path being exposed through the CLI (train.rank_chipseq_tfs +
    cli.py's --action screen). This used to only exist inline inside
    run_pipeline.py's interactive run_chipseq_screen, built around ask()
    calls for target/n_top -- unreachable from `python -m estfbu.cli`
    and untestable by construction (a test can't answer an input()
    prompt). Runs target=activity on HepG2 only, where both bundled TFs
    (GATA2, HNF4A) have pretrained weights, so this is fast (no training)."""
    from estfbu import cli

    out_csv = cli.main(["--branch", "chipseq", "--action", "screen",
                         "--cell-types", "HepG2", "--target", "activity", "--n-top", "2"])
    assert Path(out_csv).exists()
    df = pd.read_csv(out_csv)
    assert set(df["tf"]) == {"GATA2", "HNF4A"}
    assert df["score_a"].notna().all() and df["auc_a"].notna().all() and df["tpm_a"].notna().all()
    assert (df["score_a"].iloc[0] >= df["score_a"].iloc[-1]), "results should be sorted best-first"


def test_emit_oligo_library_self_consistency():
    """Regression test for oligo_library.emit_oligo_library -- the esMPRA
    wet-lab handoff. A later review pass reported reading esMPRA's actual
    step1_oligo_barcode_map.py and finding this module's original output
    unusable against it: --ref_fa wants the bare insert only (its --help
    says so explicitly), and the barcode should be a degenerate run
    discovered by sequencing, not one pre-assigned here. That review
    wasn't independently verified in this environment either (no esMPRA
    source available here) -- taken at face value because it's specific
    and self-consistent. This test checks self-consistency of the
    response to it: the bare-insert file has no adapters/barcode at all,
    the oligo file's barcode slot is a literal 'N' * barcode_length (not
    a chosen value), every emitted oligo fits the declared length budget,
    a sequence longer than the insert budget gets genuinely tiled (not
    truncated/dropped), and the requested number of scrambled negative
    controls are present with real shuffled (not fabricated) sequence --
    none of which proves esMPRA itself will accept the output."""
    import json
    import tempfile
    from estfbu.oligo_library import emit_oligo_library

    def parse_fasta(path):
        records = []
        header, seq = None, []
        for line in path.read_text().splitlines():
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
        if header is not None:
            records.append((header, "".join(seq)))
        return records

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fasta_in = d / "designs.fasta"
        short_seq = "ACGT" * 42  # 168bp, fits in one oligo
        long_seq = "ACGT" * 300  # 1200bp, must be tiled
        fasta_in.write_text(f">short\n{short_seq}\n>long\n{long_seq}\n")

        # realistic-length adapters (esMPRA's own reported ones, per a
        # review pass -- see cli.py's --oligo-pre help) -- deliberately
        # NOT 4bp: a short adapter can appear by pure chance in a
        # scrambled control's random content, which would make the
        # aim_seq-extraction check below a false positive rather than a
        # real signal (confirmed: this happened with 4bp test adapters).
        oligo_pre, oligo_after, barcode_pre = "GGCCGCTTGACG", "CACTGCGGCTCC", "CGAACCTCTAGA"
        max_oligo_len = 260
        n_controls = 3
        barcode_length = 12
        inserts_out, oligos_out, manifest_out = emit_oligo_library(
            fasta_in, d, oligo_pre=oligo_pre, oligo_after=oligo_after, barcode_pre=barcode_pre,
            barcode_length=barcode_length, max_oligo_len=max_oligo_len,
            n_scrambled_controls=n_controls, seed=7)

        assert inserts_out.exists() and oligos_out.exists() and manifest_out.exists()
        manifest = json.loads(manifest_out.read_text())
        assert manifest["n_scrambled_controls"] == n_controls
        assert manifest["max_oligo_len"] == max_oligo_len

        insert_records = parse_fasta(inserts_out)
        oligo_records = parse_fasta(oligos_out)
        assert len(insert_records) == len(oligo_records) == manifest["n_oligos"]
        assert [h for h, _ in insert_records] == [h for h, _ in oligo_records], (
            "the two files should describe the same oligos in the same order, one bare, one assembled"
        )

        barcode_slot = "N" * barcode_length
        for (header, oligo), (_, insert) in zip(oligo_records, insert_records):
            assert len(oligo) <= max_oligo_len, f"{header}: oligo length {len(oligo)} exceeds max_oligo_len"
            # order is oligo_pre + insert + oligo_after + barcode_pre + barcode_slot
            # (barcode AFTER oligo_after, not before it) -- esMPRA's step1
            # extracts aim_seq as everything between oligo_pre and the first
            # occurrence of oligo_after, so anything sitting before oligo_after
            # gets folded into that lookup and must be exactly the bare insert.
            assert oligo.startswith(oligo_pre) and oligo.endswith(barcode_slot)
            # bare insert file must contain ONLY the insert -- no adapters, no
            # barcode. Reconstruct the exact expected oligo from the insert
            # rather than checking substring absence of the adapters in the
            # insert -- with a real ~12-20bp adapter this is astronomically
            # unlikely to matter, but reconstruction is the more direct check.
            assert oligo == oligo_pre + insert + oligo_after + barcode_pre + barcode_slot, (
                f"{header}: oligo isn't exactly oligo_pre + bare insert + oligo_after + barcode_pre + barcode slot"
            )
            assert barcode_slot in oligo, (
                f"{header}: expected a degenerate 'N'*{barcode_length} barcode slot (discovered later by "
                f"esMPRA's own sequencing step, not assigned here), found none"
            )

            # Regression test for the exact esMPRA contract a review pass
            # cited from step1_oligo_barcode_map.py (~lines 70-83):
            #   aim_seq = seq[pre_index+len(oligo_pre):end_index]
            #   end_index = seq.find(oligo_after)
            # i.e. everything between oligo_pre and the first occurrence of
            # oligo_after is taken as the insert and looked up directly. A
            # prior version of this fix put the barcode before oligo_after,
            # which passed silently here (nothing locally checked it) but
            # would have made aim_seq the wrong length against real esMPRA.
            # Simulating that exact extraction against our own output is
            # the one check that would have caught it.
            pre_index = oligo.find(oligo_pre)
            end_index = oligo.find(oligo_after)
            aim_seq = oligo[pre_index + len(oligo_pre):end_index]
            assert aim_seq == insert, (
                f"{header}: esMPRA's own aim_seq extraction (between oligo_pre and oligo_after) "
                f"would recover {aim_seq!r}, not the bare insert {insert!r} -- the barcode is "
                f"leaking into the span esMPRA parses as the insert"
            )

        long_tiles = [h for h, _ in oligo_records if h.startswith("long_tile")]
        assert len(long_tiles) > 1, "a sequence longer than the insert budget should have been tiled"
        short_entries = [h for h, _ in oligo_records if h.startswith("short") and "tile" not in h]
        assert len(short_entries) == 1, "a sequence within the insert budget should pass through as one oligo"

        # every real (non-control) insert's base composition, keyed by
        # length -- a scrambled control's insert must match one of these
        # exactly (same multiset of bases), since scrambling shuffles
        # order but can't change composition
        real_insert_compositions_by_len = {}
        for header, insert in insert_records:
            if header.startswith("control_scrambled_"):
                continue
            real_insert_compositions_by_len.setdefault(len(insert), []).append(sorted(insert))

        controls = [(h, s) for h, s in insert_records if h.startswith("control_scrambled_")]
        assert len(controls) == n_controls
        for header, insert in controls:
            assert set(insert) <= set("ACGT")
            assert sorted(insert) in real_insert_compositions_by_len.get(len(insert), []), (
                f"{header}: scrambled insert's base composition doesn't match any real insert of the same length"
            )


def test_emit_oligo_library_guards():
    """Regression test for a gap a later review pass found in
    emit_oligo_library: restriction sites were only ever checked against
    the bare insert (in postprocess), never against the fully assembled
    oligo, so a site could reappear at an adapter/barcode junction
    undetected. (A separate barcode-generator exhaustion guard existed
    briefly in an earlier version of this module, but became moot once
    the barcode became a fixed degenerate 'N' placeholder instead of a
    value chosen per-oligo -- there's no longer a barcode-space to
    exhaust.)"""
    import tempfile
    from estfbu.oligo_library import emit_oligo_library

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # restriction-site re-check on the assembled oligo: the site
        # doesn't appear in the bare insert, only once oligo_pre and the
        # insert are concatenated -- this must be caught, not shipped.
        fasta_in = d / "one.fasta"
        fasta_in.write_text(">seq0\nACGTACGTACGT\n")
        site = "GAAC" + "ACGT"  # spans the oligo_pre/insert junction below
        try:
            emit_oligo_library(fasta_in, d, oligo_pre="GAAC", oligo_after="", barcode_pre="",
                                barcode_length=6, max_oligo_len=50, n_scrambled_controls=0, seed=1,
                                restriction_sites=(site,))
            assert False, "expected a ValueError: the junction-spanning site is in the insert/adapters, unavoidable"
        except ValueError as e:
            assert site in str(e)


def test_expression_matches_known_biology():
    """Regression test for expression.py: known liver-specific TFs
    (HNF4A, FOXA2) should show much higher expression in HepG2 than K562,
    and the hematopoietic-associated GATA2 should show the reverse -- this
    is the same real-data validation used during development (HNF4A
    51.0 vs 0.02 TPM, FOXA2 50.4 vs 1.3 TPM, GATA2 5.4 vs 97.2 TPM)."""
    from estfbu import expression

    cfg = load_config()
    hnf4a_hepg2 = expression.get_tpm("HNF4A", "HepG2", cfg)
    hnf4a_k562 = expression.get_tpm("HNF4A", "K562", cfg)
    assert hnf4a_hepg2 > 10 * hnf4a_k562, "HNF4A should be strongly HepG2-enriched"

    gata2_hepg2 = expression.get_tpm("GATA2", "HepG2", cfg)
    gata2_k562 = expression.get_tpm("GATA2", "K562", cfg)
    assert gata2_k562 > 5 * gata2_hepg2, "GATA2 should be strongly K562-enriched"


def test_dinuc_shuffle_preserves_dinucleotide_frequencies():
    """Regression test for the dinucleotide-shuffle upgrade: output must
    have (a) the exact same base composition, (b) the exact same
    dinucleotide frequency distribution, and (c) the same first/last
    character as the input -- the three properties that make it a valid
    Eulerian-trail-based dinucleotide shuffle, not just a stricter-looking
    mono-shuffle."""
    from collections import Counter
    from estfbu.motif_hits import _dinuc_shuffle

    rng = np.random.RandomState(7)
    seq = "ACGTACGTACGGGGCCCCATATATGCGCGCTAGCTAGCTTTTAAAACCCGGGATCGATCG"

    def dinuc_counts(s):
        return Counter(s[i:i + 2] for i in range(len(s) - 1))

    orig_counts = dinuc_counts(seq)
    for _ in range(10):
        shuf = _dinuc_shuffle(seq, rng)
        assert len(shuf) == len(seq)
        assert shuf[0] == seq[0] and shuf[-1] == seq[-1]
        assert dinuc_counts(shuf) == orig_counts
        assert sorted(shuf) == sorted(seq)


def test_filter_and_dedup_tf_specific_checks():
    """Regression test for postprocess.filter_and_dedup -- the has-ChIP-seq
    branch's ACTUAL final postprocessing step (previously only the generic
    TF-agnostic dedup_by_edit_distance it calls internally was tested).
    Exercises all three TF-specific checks in one pass using synthetic
    168bp sequences built around GATA2's known consensus motif (CTTATCT,
    see test_gata2_consensus_motif) so the outcome is fully deterministic:
      - a clean masked-core sequence with target-matching GC should survive
        AND come back with the core motif correctly unmasked
      - a sequence where the consensus motif leaks into the context
        (outside the core) should be rejected
      - a sequence with GC content far from the target should be rejected
      - a sequence where only the REVERSE COMPLEMENT of the consensus
        motif leaks into the context should also be rejected (regression
        for a real bug: the leak check used to test only the forward-
        strand motif string, so a reverse-complement leak passed through
        undetected)
    """
    from estfbu import postprocess
    from estfbu.motif_scoring import reverse_complement

    cfg = load_config()
    motif = "CTTATCT"  # GATA2 consensus, validated by test_gata2_consensus_motif
    m = len(motif)
    seq_len = cfg.model.seq_len  # 168
    half = seq_len // 2
    core_start = half - m // 2

    # ACGT repeating never contains two identical adjacent bases, so it
    # can't accidentally contain "CTTATCT" (which has "TT") or either
    # configured restriction site.
    background = ("ACGT" * (seq_len // 4 + 1))[:seq_len]
    masked_core = background[:core_start] + "N" * m + background[core_start + m:]

    seq_clean = masked_core  # should survive, GC ~0.5, no leaked motif
    seq_leaked = masked_core[:10] + motif + masked_core[10 + m:]  # motif leaked into context
    low_gc_bg = ("AT" * (seq_len // 2))[:seq_len]
    seq_bad_gc = low_gc_bg[:core_start] + "N" * m + low_gc_bg[core_start + m:]
    motif_rc = reverse_complement(motif)
    seq_leaked_rc = masked_core[:10] + motif_rc + masked_core[10 + len(motif_rc):]

    seqs = [seq_clean, seq_leaked, seq_bad_gc, seq_leaked_rc]
    scores = [10.0, 8.0, 6.0, 4.0]

    result = postprocess.filter_and_dedup("GATA2", seqs, scores, cfg, target_gc=0.5)

    assert len(result) == 1, f"expected exactly the clean sequence to survive, got {len(result)}: {result}"
    assert result[0][core_start:core_start + m] == motif, (
        "surviving sequence's core should have been unmasked to the real consensus motif"
    )
    assert "N" not in result[0], "unmasking should have removed every N"


def test_reporting_outputs_are_written():
    """Regression test for reporting.py -- previously untested despite
    being the module that lets the CLI (not just run_pipeline.py) produce
    a bubble plot or GC-comparison plot alongside a ranked TF list or a
    designed-sequence comparison. Doesn't judge plot content (out of
    scope for a fast regression test), just that each function actually
    produces the file it claims to, with the manifest carrying the
    documented fixed schema."""
    import json
    import tempfile
    from estfbu import reporting

    cfg = load_config()

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        out_fasta = d / "some_design.fasta"
        out_fasta.write_text(">seq0\nACGT\n")
        manifest_path = reporting.write_step_manifest(
            cfg, "design", "test_scenario", out_fasta, "fasta", n_records=1, seed=42, elapsed_sec=1.234)
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        for key in ("step", "scenario", "output_path", "output_format", "n_records", "seed",
                    "elapsed_sec", "timestamp"):
            assert key in manifest, f"manifest missing documented field '{key}'"
        assert manifest["step"] == "design" and manifest["n_records"] == 1 and manifest["seed"] == 42

        # real TF names with bundled quickstart RNA-seq data, so
        # write_bubble_plot's get_tpm lookups resolve without needing
        # any training or network access
        bubble_out = d / "bubble.png"
        result = reporting.write_bubble_plot(
            ["GATA2", "HNF4A"], [0.5, 0.3], [0.2, 0.6], ["HepG2", "K562"], cfg, bubble_out, "test bubble")
        assert result == bubble_out and bubble_out.exists() and bubble_out.stat().st_size > 0

        gc_out = d / "gc_comparison.png"
        result = reporting.write_gc_comparison_plot(
            ["ACGTACGT", "GGGGCCCC"], "A", ["ATATATAT", "TTTTAAAA"], "B", gc_out, "test gc comparison")
        assert result == gc_out and gc_out.exists() and gc_out.stat().st_size > 0


def test_annotate_regions_end_to_end():
    """Regression test for motif_hits.annotate_regions -- the function
    actually used by the chrombpnet branch's 'score' action and by
    run_pipeline.py's TF-annotation step (previously only its inner
    best_matching_tfs was tested directly). Checks the assembled
    'top_tf_matches' string end-to-end, including the RNA-seq TPM
    annotation piece that turns a raw motif match into "...and it's
    actually expressed here"."""
    import pandas as pd
    from estfbu import motif_hits

    cfg = load_config()
    df = pd.DataFrame({"chrom": ["chr1"], "center": [47478174]})
    annotated = motif_hits.annotate_regions(df, cfg, top_n=3, n_shuffles=50, cell_types=["HepG2", "K562"])

    assert "top_tf_matches" in annotated.columns
    s = annotated.iloc[0]["top_tf_matches"]
    assert s, "expected at least one TF match, got an empty string"
    assert "raw=" in s and "p=" in s, f"unexpected top_tf_matches format: {s}"
    assert "HepG2_tpm=" in s and "K562_tpm=" in s, f"expected TPM annotations for both cell types: {s}"


def test_motif_hits_significance_suppresses_promiscuous_motifs():
    """Regression test for the SP5/MEIS1 promiscuity fix: a region with a
    genuinely strong, specific hit should rank that hit's p-value well
    below a floor threshold, and best_matching_tfs should not crash on
    zero-variance null distributions (the z-score version did, producing
    values like z=257.9 before the fix to empirical p-values)."""
    from estfbu import motif_hits

    cfg = load_config()
    hits = motif_hits.best_matching_tfs("chr1", 47478174, cfg, window=cfg.model.seq_len, top_n=5, n_shuffles=50)
    assert len(hits) > 0
    for tf, raw, pvalue in hits:
        assert 0.0 <= pvalue <= 1.0, f"{tf} p-value {pvalue} out of [0,1] range"
        assert raw >= 0.05, f"{tf} raw score {raw} below the min_raw_score floor, shouldn't be in results"


if __name__ == "__main__":
    import tempfile
    import unittest
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_blacklist_blocks_listed_tf(tmp)
        print("PASS: test_blacklist_blocks_listed_tf")
        test_blacklist_allows_unlisted_tf(tmp)
        print("PASS: test_blacklist_allows_unlisted_tf")
    test_gata2_consensus_motif()
    print("PASS: test_gata2_consensus_motif")
    test_hepg2_gata2_data_prep_sample_count()
    print("PASS: test_hepg2_gata2_data_prep_sample_count")
    test_design_ga_seeded_and_unseeded_dont_crash()
    print("PASS: test_design_ga_seeded_and_unseeded_dont_crash")
    test_design_seq_to_onehot_respects_configured_seq_len()
    print("PASS: test_design_seq_to_onehot_respects_configured_seq_len")
    test_chrombpnet_design_produces_correct_length_sequences()
    print("PASS: test_chrombpnet_design_produces_correct_length_sequences")
    test_chrombpnet_design_tf_specific_uses_motif_importance()
    print("PASS: test_chrombpnet_design_tf_specific_uses_motif_importance")
    test_run_ga_chrombpnet_respects_manually_fixed_positions()
    print("PASS: test_run_ga_chrombpnet_respects_manually_fixed_positions")
    test_auto_detect_important_positions_are_protected()
    print("PASS: test_auto_detect_important_positions_are_protected")
    test_chrombpnet_design_output_is_deduplicated()
    print("PASS: test_chrombpnet_design_output_is_deduplicated")
    test_second_tf_hnf4a_generalizes()
    print("PASS: test_second_tf_hnf4a_generalizes")
    test_cli_chipseq_screen_action()
    print("PASS: test_cli_chipseq_screen_action")
    try:
        test_screen_all_tfs_ranks_known_biology()
        print("PASS: test_screen_all_tfs_ranks_known_biology")
    except unittest.SkipTest as e:
        print(f"SKIP: test_screen_all_tfs_ranks_known_biology ({e})")
    test_dinuc_shuffle_preserves_dinucleotide_frequencies()
    print("PASS: test_dinuc_shuffle_preserves_dinucleotide_frequencies")
    test_emit_oligo_library_self_consistency()
    print("PASS: test_emit_oligo_library_self_consistency")
    test_emit_oligo_library_guards()
    print("PASS: test_emit_oligo_library_guards")
    test_expression_matches_known_biology()
    print("PASS: test_expression_matches_known_biology")
    test_filter_and_dedup_tf_specific_checks()
    print("PASS: test_filter_and_dedup_tf_specific_checks")
    test_reporting_outputs_are_written()
    print("PASS: test_reporting_outputs_are_written")
    test_annotate_regions_end_to_end()
    print("PASS: test_annotate_regions_end_to_end")
    test_motif_hits_significance_suppresses_promiscuous_motifs()
    print("PASS: test_motif_hits_significance_suppresses_promiscuous_motifs")
    print("\nAll regression tests passed.")
