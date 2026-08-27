# Expected outputs

What each scenario actually produces, run for real against the data
bundled in `sample_data/` (via `python3 run_pipeline.py`, answering 'y' at
the first prompt). Every number below is a real measured result, not a
guess -- most scenarios were run twice to check whether the output is
bit-exact given the same seed (`training.random_seed: 42` in
`config/quickstart_config.yaml`) or has real run-to-run variance despite
that seed. See the **Determinism** note at the end for why some scenarios
land in each category.

`sample_data/` is a faster demo, not a smaller one (reduced GA iterations:
20 not 300; reduced training epochs: 15 not 1000) covering 2 of the
paper's 198 TFs (GATA2, HNF4A). The bundled ATAC/ChIP/histone bed files
are the real, full-size ENCODE peak sets, not subsampled -- so the
has-ChIP-seq branch's first step (a PPM scan over every ATAC peak) takes
just as long as on the full-scale setup. It is **not** a replacement for
the full-scale validated numbers in `ref_result/RESULTS.md` -- those used
the complete original datasets and the paper-scale hyperparameters.

Every step also writes a `<output_file>.manifest.json` alongside its real
output, in the same fixed shape regardless of scenario:
```json
{
  "step": "design", "scenario": "GATA2_HepG2_activity",
  "output_path": "...", "output_format": "fasta",
  "n_records": 100, "seed": 42, "elapsed_sec": 14.2,
  "timestamp": "2026-08-27T14:54:47"
}
```

---

## File requirements per scenario

Traced directly from the code (every `cfg.<field>` read in `run_pipeline.py`),
not inferred. `config/blacklist.txt` is used by every scenario except a
pure genome-wide screen. Paths below are `default_config.yaml`'s field
names; `quickstart_config.yaml` points the same fields at `sample_data/`.

### has-ChIP-seq branch -- base files (all 3 modes need these)

| Field | What it is | Used for |
|---|---|---|
| `genome_fasta` | hg38.fa | extracting sequence windows |
| `jaspar_pfm_cache` | jaspar_pfms.json | finding/masking the TF's core motif |
| `deeptfbu_repo` | SeqRegressionModel.py | the model architecture class itself |
| `cell_types.<ct>.atac_bed` | ATAC-seq peaks | negative-sample candidates |
| `cell_types.<ct>.chip_bed_dir` (→ `<ct>_ChIP_<tf>.bed`) | real ChIP-seq peaks | the actual binding-site positives |
| `cell_types.<ct>.h3k4me1_bed` / `h3k4me3_bed` | histone marks | GC/histone-matched negative sampling |
| `hepg2_pretrained_weights_dir` | pretrained `.pth` | **only if** that cell type's `pretrained: true` -- otherwise trains from scratch using the files above instead (no weight file needed, but ~20-70+ min, see System Requirements) |

- **Rank (no TF)**: the above, looped once per candidate TF, **plus** `tf_to_ensembl_map` + `rna_seq.<ct>.gene_quantifications_tsv` (RNA-seq TPM, printed per TF and written as `tpm_a`/`tpm_b` columns in the CSV -- not just the bubble plot's color -- so the user can check a top candidate's own gene is actually expressed before trusting the ranking); also asks up front whether to rank by activity or specificity, since ranking order is the whole point of this mode
- **Design (TF specified)**: just the base files, for that one TF. No activity/specificity question is asked -- both are run automatically (the GA needs one objective per run, so it just runs both back-to-back and hands back two FASTA files plus a GC-content comparison plot) if 2 cell types were given; with only 1 cell type, only `activity` runs (specificity needs two cell types to compare)

### no-ChIP-seq/ChromBPNet branch -- base files (all 3 actions need these)

| Field | What it is | Used for |
|---|---|---|
| `genome_fasta` | hg38.fa | extracting sequence windows |
| `cell_types.<ct>.atac_bed` | ATAC-seq peaks | which regions to score/seed from |
| `chrombpnet_models.<ct>.nobias_h5` | pretrained ChromBPNet | accessibility prediction, no training ever |

Nothing from the has-ChIP-seq branch's file set (`chip_bed_dir`, `h3k4me1/3_bed`,
`deeptfbu_repo`, pretrained weights) is touched anywhere in this branch --
that's the entire point of it.

- **Score**: base files **plus** `jaspar_pfm_cache` (every region gets scanned against all 198 TFs for annotation) and `tf_to_ensembl_map` + `rna_seq.*` (TPM shown per match); `blacklist_file` only if a TF was specified (re-ranking that one TF's motif). No activity/specificity question either: with 2 cell types it always computes both cell types' scores plus their `diff` in one pass, since scoring an already-existing region against a frozen model is cheap enough not to need a pre-pick
- **Design, no TF specified**: **only** the base files. No `jaspar_pfm_cache` -- with no TF there's no per-TF motif to condition on, so the GA optimizes raw predicted accessibility directly (ChromBPNet isn't per-TF). Same "run both, don't ask" pattern as the has-ChIP-seq branch's design mode -- two FASTA files plus a GC-content comparison plot when 2 cell types are given
- **Design, TF specified**: base files **plus** `jaspar_pfm_cache`. A specified TF switches the fitness function to a gradient-based motif-importance proxy (mean input-x-gradient importance within the TF's best PWM-match window, one backward pass per generation) instead of raw accessibility -- this is the ChIP-seq-free stand-in for the trained classifier the has-ChIP-seq branch uses, letting this mode design a TF-specific sequence without any ChIP-seq data at all (see `design.run_ga_chrombpnet`'s docstring). Same dual-objective, two-FASTA-plus-comparison-plot output shape as the no-TF case
- **Rank (screen)**: base files' `atac_bed`/`nobias_h5` were already baked into the precomputed scan (`precomputed_genome_scan`, or a from-scratch scan built the same way), so this mode instead needs that scan CSV directly, plus `jaspar_pfm_cache` (ranking all 198 TFs) and `tf_to_ensembl_map` + `rna_seq.*` (`tpm_a`/`tpm_b` columns on every row of the output CSV, not just the bubble plot's color -- same reasoning as the has-ChIP-seq rank mode above). Still asks activity-vs-specificity up front, same reasoning as the has-ChIP-seq rank mode. Not hardcoded to HepG2/K562 -- works for any two cell types a matching scan CSV was built for (see `test_scripts/chrombpnet_full_genome_scan.py <cell_type_a> <cell_type_b>`); the bundled precomputed scan only covers HepG2 vs K562, so that's the only pair runnable without a fresh multi-hour scan

A specified TF now offers either `score` or `design` in the no-ChIP-seq branch (both can use
it); `screen` is explicitly the "don't specify a TF" mode (it's what finds candidate TFs in
the first place), so the interactive script asks "Specify a TF?" first and only offers
`screen` if you say no.

### Manually fixing part of a starting sequence

Both branches' "Fixed starting sequence?" prompt accepts lowercase letters as a way to mark
specific positions that must never be mutated or recombined away for the whole GA run --
uppercase stays free to optimize, matching the common soft-masking convention (lowercase =
protected). This is the manual version of the whiteboard's "固定部分序列" case: you already
know a sub-region is important (e.g. a verified functional element inside a larger
enhancer you're improving) and want the GA to leave it alone while it works on the rest.
Deliberately manual by default -- the original design brief called auto-detecting
"important" regions optional/future-work, not needed for a first version.

In the has-ChIP-seq branch this coexists with the existing automatic TF-core-motif masking:
the TF's own motif is always protected (and gets the real consensus spliced back in during
postprocessing, same as before), and any additional lowercase-marked positions elsewhere in
the context are protected too. An all-uppercase starting sequence (the common case) behaves
exactly as before -- nothing extra is protected. In the no-ChIP-seq branch there's no
automatic masking at all, so this manual mechanism is the only way to protect part of a
starting sequence there. See `design._parse_fixed_positions` for the implementation.

**Auto-detecting important positions** (no-ChIP-seq branch's `design` action only): now
implemented too, since it turned out cheap on top of gap 1's gradient-importance machinery
rather than a separate feature. Answer 'y' to "Also auto-detect and protect the most
important existing positions?" after giving a starting sequence, and give a fraction (default
0.05 = top 5%). Ranks every position by `|gradient_importance_batch|` (magnitude, since a
position can matter by pushing the score either up or down) and protects the top fraction --
unioned with any lowercase-marked positions, not a replacement for them. Still opt-in, not
the default; manual marking stays the primary, predictable mechanism. See
`chrombpnet_scoring.auto_detect_important_positions`.

---

## 1. Blacklist gate

**Any scenario, TF already in `config/blacklist.txt`.**

Deterministic by construction (no randomness involved). The gate runs at
TF-*selection* time, not as a reactive check after a TF is chosen:

- **`run_pipeline.py` (interactive)**: a blacklisted TF is excluded from
  the candidate list before it's ever offered as an option --
  `_exclude_blacklisted()` filters `available_chipseq_tfs()` /
  `_all_tf_names()` up front, so there's nothing to reject afterward. If
  every ChIP-seq-available TF for the requested cell types happens to be
  blacklisted, the list is simply empty and the run stops with
  `[STOP] no TF has the data this run needs`.
- **CLI (`python -m estfbu.cli --tf ...`)**: since the TF is typed
  directly (no candidate list to pre-filter), the check is the first
  thing that runs, before any data prep -- exits with
  `[BLOCKED] '<TF>' is in the blacklist (...)`.

Edit `config/blacklist.txt` to change which TFs are blocked.

## 2. has-ChIP-seq branch, single cell type (activity only)

**Prompts:** `chipseq` → cell types `HepG2` → specify a TF? `y` → TF `GATA2`
→ fixed starting sequence? `n` → demo iteration count `20` → GC target `0.5`

No activity/specificity question is asked -- with only 1 cell type,
specificity has nothing to compare against, so only `activity` runs
(the script prints that explicitly and skips straight to it). Uses the
bundled pretrained HepG2 weight (no training). Measured on 2 runs,
**bit-exact identical**:

| Step | Output | Measured |
|---|---|---|
| Data prep | `data_prep/HepG2/step4_GATA2.h5` | balanced pos/neg windows (same real HepG2 ChIP-seq data as the full-scale run) |
| GA design (20 iterations) | in-memory archive | **1222 unique candidates**, best raw score **0.9992** |
| Post-processing (GC target 0.5) | `results/GATA2_HepG2_activity.fasta` | **100 sequences** |

## 3. has-ChIP-seq branch, two cell types (activity + specificity, both automatic)

**Prompts:** `chipseq` → cell types `HepG2,K562` → specify a TF? `y` → TF
`GATA2` → fixed starting sequence? `n` → demo iteration count `20` → GC
target `0.5`

With 2 cell types given, both `activity` and `specificity` design runs
happen automatically, back to back, in the same run -- no upfront
question asks you to pick one first (the GA needs one objective per
run, so instead of guessing, this mode just runs both and hands back
both FASTA outputs plus a GC-content comparison plot so you can compare
visually). HepG2 uses the pretrained weight; **K562 trains from
scratch** on the bundled data -- and *then a second model trains from
scratch too* (the HepG2-vs-K562 comparative model the specificity run
needs). This is the one scenario in the whole pipeline that touches
real from-scratch training, and the one place a GPU actually matters --
see **System requirements** in the README.

**Cost, this machine:** CPU-only, a single epoch of K562's from-scratch
train measured over 70 minutes -- consistent with DeepTFBU's own README
warning about this exact architecture without a GPU. `train_context_model`
automatically uses MPS (Apple Silicon GPU) when available; with that,
training both models from scratch (K562-alone, then HepG2-vs-K562
comparative) took **~24 min + ~14 min ≈ 38 min real wall-clock time**.
Still the slowest scenario in this document by far, but a very different
story than CPU alone.

| Run | K562 train + comparative train + GA (wall-clock) | Unique candidates | Best raw score | Final sequences |
|---|---|---|---|---|
| 1 (fresh) | ~38 min total (~24 min K562 + ~14 min comparative+GA) | **1229** | **1.8717** | **100** |
| 2 (fresh, independent retrain) | ~38 min total (~23 min K562 + ~15 min comparative+GA) | **1215** | **1.7858** | **100** |

Both runs used the same seed (42) and are genuinely independent
from-scratch retrains (both models' caches cleared before run 2, not a
cache reuse) -- this is the real measured variance, not a guess:

- **Unique candidates**: 1229 vs 1215 (**~1.1% spread**)
- **Best raw score**: 1.8717 vs 1.7858 (**~4.6% relative spread**)
- **Final sequence count**: 100 both times (the postprocessing cap;
  both runs had comfortably more than 100 candidates survive filtering)
- **Designed sequences themselves differ** between runs (confirmed via
  diff -- expected, since the two trained models are numerically
  different, not just differently-ordered)

**Practical read**: expect similar-magnitude output (roughly a thousand-
plus candidates, best score in the high-1s to ~1.9 range, 100 final
sequences) but not identical numbers, if you re-run this scenario
yourself. The GC target and other config values are what stayed fixed;
the trained model itself is what varies.

The activity run (same 100-cap postprocessing) writes alongside it, from
the same `GATA2, HepG2+K562` run:
`results/GATA2_HepG2_K562_activity.fasta`,
`results/GATA2_HepG2_K562_specificity.fasta`, and
`results/GATA2_HepG2_K562_activity_vs_specificity_gc.png` (the GC-content
comparison plot -- the visual check that replaced asking which objective
you wanted in advance).

**Mechanism** (why any variance here differs from every other scenario):
even with every RNG seeded (`seed_everything`, seeded `DataLoader`
generators -- see Determinism note), floating-point summation isn't
strictly associative under multi-threaded/GPU execution, so
conv/BatchNorm ops can accumulate in a slightly different order between
runs. Early stopping (`patience: 5`) can amplify a tiny numeric
difference into a different number of epochs trained, and from there a
different final model -- this is a property of the hardware/parallelism,
not a bug in the seeding.

## 4. no-ChIP-seq/ChromBPNet branch, `score` action

**Prompts:** `chrombpnet` → specify a TF? `n` → action `score` → cell
types `HepG2,K562` → max regions `50` → annotate top `3`

No target question: with 2 cell types given, this always computes both
cell types' scores plus their `diff` in the same pass (a specified TF
would instead lock the action to `score` and add a re-ranking step by
that TF's own motif -- not exercised in this scenario). Pure inference
against frozen pretrained ChromBPNet models -- no training, no
model-side randomness. Measured on 2 runs, **bit-exact identical**:

| Step | Output | Measured |
|---|---|---|
| Primary scan (50 HepG2 ATAC peaks) | in-memory | HepG2 ChromBPNet scores, e.g. top region `chr1:912689` = 6.6376 |
| Differential scan (+K562 model) | `results/chrombpnet_HepG2_K562.csv` | 50 regions, top `diff` = **2.1758** at `chr1:912689` |
| TF annotation (top/bottom 3) | `results/chrombpnet_HepG2_K562_annotated_top3.csv` | e.g. top region's best match: `IKZF1(raw=0.37,p=0.010,HepG2_tpm=0.0,K562_tpm=35.4)` |

Elapsed (score step, this run size): **38.7s**.

## 5. no-ChIP-seq/ChromBPNet branch, `design` action

**Prompts:** `chrombpnet` → specify a TF? `n` → action `design` → cell
types `HepG2,K562` → fixed starting sequence? `n` → use config's 20
iterations (don't reduce further)

Same "run both, don't ask" pattern as the has-ChIP-seq branch's design
mode: no target question, both `activity` and `specificity` design runs
happen automatically since 2 cell types were given, each producing its
own FASTA plus one shared GC-content comparison plot. Also pure
inference against frozen pretrained models (the GA only mutates the
input sequence, never the model). Measured on 2 runs, **bit-exact
identical** per objective: **1236 unique candidates**, best raw score
**3.4142**, **100 final sequences** after dedup (this is the
`specificity` run's numbers; `activity` runs alongside it with its own
counts). Elapsed: **146s per objective**.

Outputs: `results/chrombpnet_design_HepG2_K562_activity.fasta`,
`results/chrombpnet_design_HepG2_K562_specificity.fasta`,
`results/chrombpnet_design_HepG2_K562_activity_vs_specificity_gc.png`.

## 6. no-ChIP-seq/ChromBPNet branch, `screen` action

**Prompts:** `chrombpnet` → `screen` → cell types `HepG2,K562` → target
`specificity`

Uses the bundled precomputed genome-wide scan
(`sample_data/precomputed/chrombpnet_full_genome_scan_hepg2_atac_peaks.csv`,
auto-copied into place) instead of re-running the ~90min full scan. This
file is **byte-identical** to the one behind `ref_result/RESULTS.md`'s
genome-wide screening result, and the ranking itself is pure motif
math (no model, no RNG) -- so this scenario's output is deterministic and
already validated:

| Rank | TF | mean_diff_strong | mean_diff_rest | p-value | tpm_a (HepG2) | tpm_b (K562) |
|---|---|---|---|---|---|---|
| 1 | KLF15 | 0.0324 | 0.2780 | <1e-300† | 0.91 | 0.83 |
| 2 | RXRB | 0.4888 | 0.2271 | <1e-300† | NaN | NaN |
| 3 | HNF4A | 0.5861 | 0.2156 | <1e-300† | 51.01 | 0.02 |
| 4 | HNF4G | 0.6721 | 0.2067 | <1e-300† | 31.41 | 0.00 |
| 5 | FOXA1 | 0.6265 | 0.2117 | <1e-300† | 60.32 | 0.90 |

†The CSV's raw `pvalue` column literally stores `0.0` for these rows --
not a rounded display, a genuine floating-point underflow. With
16,000+ regions per group and this much separation, scipy's
`mannwhitneyu` normal-approximation z-score is large enough that the
true p-value (astronomically small, plausibly far below 1e-300) falls
below the smallest representable `float64` (~5e-324) and collapses to
exactly 0.0. `<1e-300` here means "known to be smaller than that,"
not "computed to be 1e-300" -- the actual value isn't representable.

**Why the TPM columns matter here, concretely**: a low, non-significant p-value
from the motif/accessibility test alone doesn't mean a TF is actually the
right biological answer. HNF4A/HNF4G/FOXA1 are exactly the well-known
hepatocyte master regulators the highlight result in the README is built
on -- their TPM confirms real, strongly HepG2-specific expression, so
their high statistical rank is corroborated by independent RNA-seq data.
KLF15's motif/accessibility signal is just as statistically significant,
but its TPM is low and nearly identical between both cell types (0.91 vs
0.83) -- expression data gives no support for it being a real
cell-type-specific regulator here, exactly the kind of case this column
exists to let a user catch before trusting the ranking. RXRB's TPM is
`NaN` in both cell types -- investigated directly against Ensembl's own
REST API: `tf_to_ensembl.json`'s ID for RXRB (`ENSG00000206289`) is
correct (confirmed via `lookup/id` and `xrefs/symbol`), it's just that
RXRB's sole current Ensembl gene model sits on an MHC alternate-haplotype
scaffold (`HSCHR6_MHC_QBL_CTG1`), not the primary GRCh38 assembly -- no
primary-assembly gene ID for RXRB exists to map to instead. ENCODE's
standard RNA-seq quantification only covers the primary assembly, so this
can't be fixed by correcting the mapping. 6 of the 198 TFs in
`tf_to_ensembl.json` are affected the same way (checked all of them):
`RXRB`, `IRF9`, `MEF2D`, `PBX2`, `TFE3`, `ZNF707` -- every one's only
Ensembl gene model is on an alt-haplotype/patch scaffold. A genuine,
disclosed data limitation, not silently treated as "zero expression."

(Full 198-TF ranking: `ref_result/tf_screen_specificity_hepg2_vs_k562.csv`,
same underflow-to-0.0 caveat applies to any row showing `pvalue == 0.0`.
Regenerated against the current code: includes `tpm_a`/`tpm_b` and an
`effect_direction` column, and is sorted by signed effect size rather than
raw p-value, so the top rows are genuinely the strongest HepG2-favoring
hits, not just the largest-magnitude hits regardless of direction.
`screen_all_tfs` now takes `cell_types` explicitly instead of assuming
HepG2/K562 -- this scenario's numbers are unaffected, this is the same
HepG2-vs-K562 scan as before, just no longer hardcoded to be the only pair
the code can express.)

## 7. has-ChIP-seq branch, no TF specified (rank)

**Prompts:** `chipseq` → cell types `HepG2` → specify a TF? `n` → target
`activity` → keep top `2`

Loops both quickstart TFs (GATA2, HNF4A -- the only two with bundled
HepG2 ChIP-seq data), scoring each on its own real ChIP-seq-derived
positive windows via its own pretrained context model, same mechanism as
the has-ChIP-seq design path just without a chosen TF. Uses the bundled
pretrained HepG2 weights for both TFs (no training):

| Rank | TF | score_a (HepG2) | tpm_a (HepG2) |
|---|---|---|---|
| 1 | GATA2 | 0.5811 | 5.37 |
| 2 | HNF4A | 0.1980 | 51.01 |

Output: `results/chipseq_screen_activity_HepG2.csv` (columns `tf, score_a,
score_b, tpm_a, tpm_b` -- `score_b`/`tpm_b` empty here since this run is
single-cell-type (`target=activity`, only HepG2 given); both only
populate for `target=specificity` with 2 cell types). Within this one
cell type, ranking by model score vs ranking by TPM disagree (GATA2
scores higher on its own classifier despite much lower expression in
HepG2 than HNF4A) -- a concrete example of why the TPM column exists: a
high score from a TF's own trained context-model classifier reflects "how
strongly do this TF's real ChIP-seq positive windows fit its own model,"
not "is this TF's gene highly expressed in this cell type." Related
questions, not the same one, and this scenario shows they can point
different directions even with no second cell type involved at all.

Elapsed this run: **539s for both TFs** -- unusually slow (a normal run of
this scenario should take well under a minute per TF; see the
**Determinism** note below on why timing, not correctness, varied here).

## 8. no-ChIP-seq/ChromBPNet branch, `design` action, TF specified (gap-1 fix)

**Prompts:** `chrombpnet` → specify a TF? `y` → TF `GATA2` → action
`design` → cell types `HepG2,K562` → fixed starting sequence? `n` → demo
iterations `20`

Exercises the ChIP-seq-free TF-specific design path added this session
(`design.run_ga_chrombpnet`'s `tf=` argument): fitness is mean
input-x-gradient importance within GATA2's best PWM match, not raw
ChromBPNet accessibility. Both objectives run automatically (same "run
both, don't ask" pattern as every other design mode):

| Target | Unique candidates | Best raw score | Final (after dedup) | Elapsed |
|---|---|---|---|---|
| activity | 1235 | 0.1626 | 100 | 153.9s |
| specificity | 1242 | 0.1317 | 100 | 309.5s |

Outputs: `results/chrombpnet_design_GATA2_HepG2_K562_activity.fasta`,
`results/chrombpnet_design_GATA2_HepG2_K562_specificity.fasta`,
`results/chrombpnet_design_GATA2_HepG2_K562_activity_vs_specificity_gc.png`.
The raw scores here (importance-based) aren't directly comparable in
magnitude to scenario 5's no-TF scores (raw accessibility, log-counts
scale) -- different fitness function, different units, same output shape.

## 9. no-ChIP-seq/ChromBPNet branch, `score` action, TF specified (motif re-ranking)

**Prompts:** `chrombpnet` → specify a TF? `y` → TF `GATA2` → action
`score` → cell types `HepG2,K562` → max regions `30` → annotate top `N`

Same base differential scan as scenario 4 (30 HepG2 ATAC peaks, both cell
types' ChromBPNet scores -- top region unchanged: `chr1:912689`,
`diff=2.1758`, confirming this re-ranking step doesn't alter the base
scan), but with a TF given, the regions are then re-ranked by GATA2's own
PWM motif-match strength instead of `diff`. The re-ranked top region
differs from the diff-ranked top region (`chr1:931752`, `motif_score=
0.0673` vs `chr1:912689`) -- expected: "most differentially accessible"
and "strongest GATA2 motif match" are independent questions, and this
scenario's whole point is answering the second one instead of the first.
Elapsed: **3.5s** for the base scan (fast, pure inference against frozen
models); re-ranking itself is a numba-JIT'd PWM scan, negligible added
time.

## 10. Manually fixing part of a starting sequence (gap-2 fix)

**Prompts:** any design action, either branch → fixed starting sequence?
`y` → paste a sequence with some positions in **lowercase** → rest of the
prompts as usual

Demonstrated here on the no-ChIP-seq branch, no TF, `activity` target,
seeded from a real HepG2 ATAC peak with its first 50bp marked fixed
(lowercased):

```
fixed region (first 50bp, as given): tcctcaccctcacacctcaccctcacccaaaccataatccctaaccccta
```

| Metric | Value |
|---|---|
| Archived unique sequences | 1020 |
| Best raw score | 6.0969 |
| Sequences with the fixed region altered | **0 / 1020** |
| Elapsed | 69.8s |

Zero violations across every sequence the GA ever produced this run --
the lowercase-marked region is genuinely held constant for the entire
search, not just in the final output. This is the general mechanism;
`_seed_from_sequence` (has-ChIP-seq branch) additionally always protects
the specified TF's own core motif on top of whatever the user marks, and
remaps marked positions to the correct coordinates if the motif search
picks the reverse strand -- see `design._parse_fixed_positions` and
`design._seed_from_sequence`'s docstrings.

## 11. Wet-lab handoff: esMPRA oligo library

**Prompts:** any design run, either branch → after the design completes,
"Also emit an esMPRA-shaped oligo library?" `y` → adapters/barcode
parameters (or `--emit-oligo-library --oligo-pre ... --oligo-after ...
--barcode-pre ... --barcode-length ...` on the CLI)

This is the terminal "wet experiment" node from the original design
brief, not an optional add-on -- see the pipeline figure. Demonstrated
here on the has-ChIP-seq branch's 100 GATA2/HepG2/activity designs from
scenario 2, run through `emit_oligo_library` with a 230bp total length
budget, a 20bp adapter pair, a 4bp barcode spacer, and a 12bp barcode:

| Metric | Value |
|---|---|
| Source designs | 100 |
| Insert budget (max_oligo_len minus adapters/spacer/barcode) | 178bp |
| Oligos emitted | 105 |
| Scrambled negative controls | 5 |

105 oligos from 100 source designs, not 100: each 168bp design fits
within the 178bp insert budget in one piece (no tiling needed at this
length -- tiling only kicks in for the no-ChIP-seq branch's 2114bp
ChromBPNet designs, which don't fit any realistic array-synthesis budget
unmodified), so the difference is exactly the 5 scrambled controls.
Output: `{name}_oligo_library_inserts.fasta` (bare inserts, for esMPRA's
`--ref_fa`), `{name}_oligo_library.fasta` (array-order oligos, adapters +
a degenerate `'N' * barcode_length` slot where the physical barcode
goes), and `{name}_oligo_library.manifest.json`, alongside the base
design FASTA.

**Verified against esMPRA's real source** (github.com/WangLabTHU/esMPRA,
`step1_oligo_barcode_map.py`), not just its documented CLI shape:
`--ref_fa` genuinely wants bare inserts (its own `--help` says so
verbatim), the insert is parsed as everything between `oligo_pre` and the
first `oligo_after` match (so the barcode has to sit outside that span),
and a design only counts as usable once `--min_barcode_per_oligo`
(esMPRA's default: 3) distinct barcodes are recovered from real
sequencing reads -- which is why the barcode here is a degenerate
placeholder rather than a value assigned in silico.
`test_emit_oligo_library_self_consistency` includes a direct regression
test that simulates esMPRA's own `aim_seq` extraction against this
module's output and confirms it recovers the exact bare insert. Still not
a guarantee esMPRA will accept the output end-to-end (that needs an
actual run against esMPRA, not simulated extraction) -- if its real
behavior differs from what's confirmed here, `src/estfbu/oligo_library.py`
is the one file to fix.

---

## Determinism: what's seeded, and what still varies

`training.random_seed: 42` (quickstart config) seeds `random`, `numpy`,
and `torch`'s global RNGs once at startup (`seed_everything()` in
`run_pipeline.py`), plus every `DataLoader(shuffle=True)` gets its own
seeded generator. Given that:

- **Scenarios with no from-scratch training** (2, 4, 5, 6, 7, 8, 9, 10
  above) are bit-exact reproducible on the same machine: GA
  mutation/recombination draws from an explicitly-seeded
  `np.random.RandomState`, and model inference against an already-trained/
  pretrained frozen model has no remaining randomness to vary. Scenarios
  7-10 (added along with the gap-1/gap-2 fixes) were each only run once
  to capture the numbers documented above, not run twice like 2-6 were --
  same underlying mechanism, so bit-exact repeats are expected, just not
  independently re-confirmed here. Scenario 7's unusually long elapsed
  time (539s for two pretrained-model inference passes that would
  normally take well under a minute) reflects heavy concurrent load on
  the development machine at the time it was captured, not the pipeline
  itself -- rerun it yourself if you want a representative timing number.
- **Scenario 3 (from-scratch training)** is the one place true run-to-run
  variance is possible even with every RNG seeded: floating-point
  summation isn't strictly associative under multi-threaded/GPU
  execution, so conv/BatchNorm operations can accumulate in a slightly
  different order between runs, producing tiny numerical differences
  that early stopping can amplify into a different number of epochs
  trained, and from there a different final model. Measured real spread
  (see scenario 3 above, two independent from-scratch runs): ~1% in
  candidate count, ~5% in best raw score. Not a bug in the seeding --
  a property of parallel floating-point execution.
- **Bit-exact reproduction across *different* machines** isn't claimed
  even for the deterministic scenarios above -- different CPU
  architectures/thread counts can still produce different floating-point
  rounding. The numbers in this document are what this pipeline produces
  on the machine it was developed on; expect exact agreement on repeat
  runs on your own machine, but treat cross-machine agreement as "very
  close" rather than guaranteed bit-identical.
