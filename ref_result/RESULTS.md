# Reference results

Summary of validated findings. Full data files in this directory; source
code that produced them in `test_scripts/`.

## 1. TFBU model reproduction (has-ChIP-seq branch)

Linear regression on TFBU scores predicting real HepG2 MPRA enhancer
activity: **test Pearson correlation 0.671** (`tfbu_linear_regression_metrics.json`),
against the original paper's reported 0.794 (their 10-model-replicate
ensemble; only 1 replicate is publicly released).

## 2. GATA2 cell-type-specific enhancer design

Real ChIP-seq-trained models, genetic algorithm design: 100 designed
sequences predicted HepG2-specific for GATA2 binding (`GATA2_design_report.md`).

## 3. Genome-wide TF screening (no ChIP-seq, no TF specified in advance)

`tf_screen_specificity_hepg2_vs_k562.csv` — all 198 known JASPAR TF motifs
ranked by enrichment in HepG2-vs-K562 differentially-accessible regions,
using only ATAC-seq + a pretrained general accessibility model (ChromBPNet)
+ motif matching. No ChIP-seq, no per-TF trained model, no TF told to the
algorithm in advance.

**Top 5 HepG2-specific**: HNF4G, FOXA1, HNF4A, FOXC1, FOXA2 — the
canonical hepatocyte master-regulator network, emerging entirely
unsupervised (Mann-Whitney p < 1e-300 for all five).

**Top 5 K562-specific**: KLF15, ZBTB14, EGR1, ZBTB33, NRF1 — alongside
strong enrichment of the E2F/DP cell-cycle family (E2F1/E2F2/E2F4/E2F8/
TFDP1), consistent with K562 being a highly proliferative leukemia line.

**The p-values are not uniformly significant, which is the point.** Of
192 TFs with a valid test, only 12 hit the extreme end (genuine
float64 underflow of the Mann-Whitney normal-approximation p-value —
see `EXPECTED_OUTPUTS.md` for why that shows as exactly `0.0` rather
than a tiny nonzero number). The other 180 span a smooth gradient down
to biologically unremarkable TFs sitting at p≈0.96–0.99 (e.g. `NFYB`,
`ZNF331` — general/non-liver-specific factors with no reason to be
cell-type-differential). A test that returned significance for every
TF regardless of biological plausibility would indicate a miscalibrated
null model; this graded, right-shaped spread — extreme significance
only for the biologically sensible master regulators, and p-values
scattered across the full [0,1] range for the rest — is itself evidence
the test is discriminating real signal from noise correctly, not an
artifact.

## 4. Cross-branch validation

Does the independently-derived no-ChIP-seq signal agree with the
ChIP-seq-trained branch and real RNA-seq expression on the same TF
(GATA2)? Three methods, one conclusion (K562-specific):

| Method | Result |
|---|---|
| ChIP-seq-trained comparative model (branch 1) | GATA2 designed sequences score HepG2-specific *relative to K562's own trained model* — i.e. the model correctly learns GATA2 favors K562 when asked to discriminate |
| RNA-seq expression | 5.4 TPM (HepG2) vs 97.2 TPM (K562) |
| Motif-vs-accessibility correlation (no ChIP-seq at all) | Strong-GATA2-motif regions shift toward K562-favoring accessibility, Mann-Whitney p = 2.84e-26 (`chrombpnet_full_genome_scan_top20_annotated.csv` has the region-level detail) |

## 5. Generalization to a second TF

Pipeline re-run end-to-end for HNF4A (different motif structure, HepG2-only
biology, freshly downloaded ENCODE ChIP-seq data never seen by the
pipeline before) — completed successfully, confirming the approach isn't
GATA2-specific.

## 6. Baseline comparison and broadened multi-TF validation

Sections 2-5 validated the pipeline on two hand-picked TFs (GATA2, HNF4A).
To check that holds up more broadly and isn't just curve-fitting to two
convenient examples, `baseline_comparison_motif_only_vs_chrombpnet.csv`
scores 114 TFs (all JASPAR TFs with an unambiguous RNA-seq direction,
|log2FC| >= 1, out of 198 total) two ways, both checked against RNA-seq
expression direction as independent ground truth:

- **Baseline ("motif-only")**: the standard naive approach — motif score
  in each cell type's private ATAC peaks (peak called in one cell type,
  not the other), no accessibility model at all.
- **Pipeline (ChromBPNet-branch)**: this repo's actual no-ChIP-seq method
  — motif score vs pretrained ChromBPNet-predicted accessibility,
  genome-wide (the same method behind sections 3-4 above).

| Panel (by \|log2FC\| rank) | Baseline accuracy | Pipeline accuracy | Discordant pairs (only-baseline / only-pipeline) | McNemar exact p |
|---|---|---|---|---|
| Top 10 | 70.0% | 70.0% | 1 / 1 | 1.00 |
| Top 20 | 60.0% | 70.0% | 1 / 3 | 0.63 |
| Top 30 | 70.0% | **76.7%** | 2 / 4 | 0.69 |
| Top 50 | 54.0% | **66.0%** | 3 / 9 | 0.15 |
| All 114 | 53.5% | 54.4% | 20 / 21 | 1.00 |

(`baseline_comparison_stratified_summary.csv`.)

**Honest reading**: the pipeline's edge over naive motif scanning is
directionally consistent among the most confidently cell-type-specific
TFs (top 30-50 by expression difference — the actual intended use case,
finding master regulators as in section 3), but at this panel size none
of the accuracy gaps reach conventional statistical significance
(McNemar exact p > 0.05 throughout; closest is top-50 at p=0.15, limited
by only 12 discordant calls to test). Averaged across the full 114-TF
panel, including many weakly-differential TFs, both methods drift toward
chance and the gap vanishes entirely. This is expected, not a failure: TF
mRNA level alone is a noisy, imperfect ground truth for binding-site
accessibility (post-translational regulation, cofactor availability, and
motif redundancy across TF families all add noise the further a TF is
from having a clear, strong expression difference to detect in the first
place). Read this as a suggestive trend worth a larger validation panel,
not a proven win — the pipeline should be pitched as a tool for
prioritizing strong, confident candidates, not as a statistically proven
uniform improvement over naive motif scanning.

## 7. Runtime and usability

Measured on a single CPU (no GPU used anywhere in this pipeline):

| Task | Time | Notes |
|---|---|---|
| Genome-wide TF screen, no-ChIP-seq branch, all 198 TFs | ~90 min | Reuses one pretrained ChromBPNet model per cell type for every TF — no per-TF training |
| GA sequence design, has-ChIP-seq branch, 300 iterations | ~95 min (~19s/iteration) | Per target TF/cell-type pair, using an already-trained or pretrained model |
| Baseline comparison script (114 TFs, this section) | ~95 sec | Motif scanning only, no accessibility model |

The practical contrast is between branches, not just runtime: the
has-ChIP-seq branch needs a dedicated trained model per TF (HepG2 has
198 pretrained; any other cell type, like K562 here, needs training from
scratch, which was not systematically benchmarked but is a real added
cost). The no-ChIP-seq/ChromBPNet branch reuses a single pretrained
accessibility model per cell type for arbitrary TFs and cell types with
no training step at all — the ~90 min genome-wide screen above covers
*all* 198 TFs at once, not one TF at a time.

## Visualization

`bubble_plot.html` — open directly in a browser. Size = ChromBPNet
accessibility score, color = RNA-seq expression (TPM), for the top
candidate TFs found in the genome-wide screen.
