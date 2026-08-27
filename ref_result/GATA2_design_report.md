# GATA2 HepG2-Specific Enhancer Design (CSpTFBU Reproduction)

Reproduction of the paper's "CSpTFBU" (cell-type-specific TFBS-context) module:
designing enhancer sequences built around a GATA2 core motif that are
predicted to be active in HepG2 but inactive in K562.

## 1. What this required (full pipeline, no shortcuts)

Unlike the main TFBU modelling task, no pretrained weights existed for the
K562-specific or HepG2-vs-K562-comparative models — only the HepG2-only
model (used in the main pipeline) was on Zenodo. Everything else was built
from scratch using the DeepTFBU repo's own bundled GATA2 ChIP-seq/ATAC-seq/
histone data and code:

1. Downloaded the hg38 reference genome (UCSC, ~980MB compressed).
2. Generated HepG2 GATA2 training data (21,001 balanced pos/neg 168bp
   windows) from bundled `HepG2_ChIP_GATA2.bed` + ATAC + histone marks.
3. Generated K562 GATA2 training data (35,833 balanced windows) the same way
   from bundled K562 data.
4. Generated a matched HepG2-vs-K562 comparative dataset (20,121 pairs).
5. **Trained two new DenseNet+BiLSTM classifiers from scratch, on CPU**
   (no pretrained weights existed for these):
   - K562 GATA2 context model — early-stopped at epoch 33, **AUC 0.7108**
   - HepG2-vs-K562 GATA2 comparative model — early-stopped at epoch 41,
     **AUC 0.7148** (best epoch 34)
   - For comparison, the paper's own reported AUC for the pretrained HepG2
     GATA2 model is **0.7104 ± 0.0093** — both from-scratch models land
     right on the paper's own quality bar for this architecture/task type.
6. Ran the paper's genetic algorithm (300 iterations, population 2000,
   30% elite/30% new per generation) combining all three models:
   `MetricSeq = Score_HepG2vsK562 + Score_HepG2 - Score_K562`
7. Post-processed: filtered out sequences containing the literal GATA2
   consensus motif embedded in the context (to keep this a genuine
   *context* effect, not just a stronger core motif), matched background
   GC-content to the training distribution (within 10%), removed
   restriction-enzyme cut sites, and deduplicated by Levenshtein edit
   distance (≥50) to keep the top 100 diverse designs.

Scripts adapted from the authors' original CUDA-only code: removed all
`.cuda()` calls, set `weights_only=False` for `torch.load` (same
provenance/trust reasoning as the main pipeline — these are either the
paper's own Zenodo weights or models we trained ourselves in this session).

## 2. Result

**34,030 sequences** passed the initial GC/motif-exclusion filter;
**100 final designs** survived deduplication. Top-100 score statistics:

| Score component | Range across top 100 |
|---|---|
| HepG2-vs-K562 score | 0.9975 – 0.9999 |
| HepG2 score | 0.9897 – 0.9996 |
| K562 score | 0.0645 – 0.0782 |

All 100 final sequences are consistently and strongly predicted to be
**active in HepG2 and inactive in K562** — the HepG2/HepG2-vs-K562 scores
are all pinned near 1.0 and the K562 score near 0.07, a >13x separation.

**Top design** (highest combined score, 1.9332):
```
TGAAGGCCATGGACCACGGAACCTGGAGTACTGGGTCCATTGCGTTCCATACCGGACCTTAGTAAACGT
AAACAATATTTACTTATCTTTAATATTTATATAGTATAGTAGTATTTAATAAATATTTACTAGGTCCA
AAGTAGGGTAAACTTTGAACTGGACTGACCA
```
(168bp, GATA2 core motif `CTTATCT` embedded at center, masked/unmasked per
the paper's windowing convention.)

## 3. Files

- `GATA2_HepG2_specific_enhancers.fasta` — final 100 designed sequences,
  headers give the combined optimization score
- `GATA2_HepG2_specific_candidates_top500.txt` — top 500 pre-dedup
  candidates with full score breakdown (columns: combined score, sequence,
  HepG2-vs-K562 score, HepG2 score, K562 score, GATA2 motif, GC content)

## 4. Caveat

Same as the main pipeline: these are one-replicate models (not the paper's
10-replicate ensemble), trained on a single TF (GATA2) using the bundled
demo data. The strength and consistency of the specificity signal (13x
separation, tight score ranges) suggests the pipeline is working correctly,
but these designs have not been experimentally validated — they are
computational predictions only, exactly as in the paper prior to its own
MPRA validation step.
