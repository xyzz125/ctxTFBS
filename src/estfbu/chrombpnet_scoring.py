"""No-ChIP-seq branch: score genomic regions / arbitrary sequences for
predicted chromatin accessibility using pretrained ChromBPNet models
(Kundaje lab, ENCODE project -- same ATAC-seq experiments the paper itself
used, see config.yaml chrombpnet_models cell_type entries).

Uses bpnet-lite (PyTorch) to load the .h5 model files directly -- no
TensorFlow/conda environment needed, keeping this consistent with the rest
of the pipeline's pure-PyTorch stack.

This module answers the whiteboard's "no ChIP-seq" branch question: given
only ATAC-seq peaks (no TF-specific ChIP-seq), which regions look most
"accessible"/TF-binding-supportive, and does that differ between cell
types? It does NOT identify which specific TF is responsible (that needs
either ChIP-seq, or a separate motif-hit-calling step layered on top --
see design.py's docstring / README for that gap).
"""
from pathlib import Path

import numpy as np
import torch
from pyfaidx import Fasta

from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks

INPUT_LEN = 2114  # standard ChromBPNet input window
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

_model_cache = {}
_genome_cache = {}


def _genome(cfg):
    path = cfg.genome_fasta
    if path not in _genome_cache:
        _genome_cache[path] = Fasta(path)
    return _genome_cache[path]


def load_model(cell_type: str, cfg):
    """Load (and cache) the pretrained ChromBPNet nobias model for a cell
    type. Requires cfg.chrombpnet_models.<cell_type>.nobias_h5 to be set."""
    if cell_type in _model_cache:
        return _model_cache[cell_type]
    from bpnetlite import BPNet
    cb = getattr(cfg, "chrombpnet_models", None)
    if cb is None or not hasattr(cb, cell_type):
        raise ValueError(
            f"No chrombpnet_models.{cell_type} entry in config. Add the path to that "
            f"cell type's downloaded model.chrombpnet_nobias.*.h5 file."
        )
    path = getattr(cb, cell_type).nobias_h5
    model = BPNet.from_chrombpnet(path)
    model.eval()
    _model_cache[cell_type] = model
    return model


def _seq_to_onehot(seq: str) -> np.ndarray:
    arr = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq):
        idx = IDX.get(c)
        if idx is not None:
            arr[0, idx, i] = 1
    return arr


def score_region(chrom: str, center: int, cell_type: str, cfg) -> float:
    """Predicted log(counts) accessibility score for a region centered at
    `center` (genomic coordinate), for one cell type."""
    model = load_model(cell_type, cfg)
    genome = _genome(cfg)
    half = INPUT_LEN // 2
    seq = genome[chrom][center - half:center + half].seq.upper()
    if len(seq) != INPUT_LEN:
        return float("nan")
    x = torch.tensor(_seq_to_onehot(seq))
    with torch.no_grad():
        _, log_counts = model(x)
    return float(log_counts.item())


def score_regions_bulk(regions, cell_type: str, cfg, batch_size: int = 32):
    """regions: list of (chrom, center) tuples. Returns np.ndarray of
    predicted log(counts) scores, same order, batched for speed."""
    model = load_model(cell_type, cfg)
    genome = _genome(cfg)
    half = INPUT_LEN // 2
    scores = np.full(len(regions), np.nan)

    batch_seqs, batch_idx = [], []

    def flush():
        if not batch_seqs:
            return
        x = torch.tensor(np.concatenate([_seq_to_onehot(s) for s in batch_seqs], axis=0))
        with torch.no_grad():
            _, log_counts = model(x)
        for j, i in enumerate(batch_idx):
            scores[i] = log_counts[j].item()
        batch_seqs.clear()
        batch_idx.clear()

    for i, (chrom, center) in enumerate(regions):
        seq = genome[chrom][center - half:center + half].seq.upper()
        if len(seq) != INPUT_LEN:
            continue
        batch_seqs.append(seq)
        batch_idx.append(i)
        if len(batch_seqs) >= batch_size:
            flush()
    flush()
    return scores


def scan_atac_peaks(cell_type: str, cfg, max_regions: int = None):
    """Score every peak in this cell type's ATAC bed with its own
    ChromBPNet model. Returns (chrom, center, score) rows. This is the
    "genome-wide scan restricted to accessible regions" the whiteboard
    describes -- scoring the whole genome base-by-base is what ChromBPNet
    is designed for, but scanning ATAC peaks only is the practical
    approximation used here (peaks are where accessibility signal exists at
    all; scoring closed chromatin elsewhere is not informative for this
    question)."""
    ct = cfg.cell_type(cell_type)
    regions = []
    with open(ct.atac_bed) as f:
        for line in f:
            parts = line.split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            regions.append((chrom, (start + end) // 2))
            if max_regions and len(regions) >= max_regions:
                break
    scores = score_regions_bulk(regions, cell_type, cfg)
    return regions, scores


# ---------------------------------------------------------------------------
# TF-specific design without ChIP-seq: the has-ChIP-seq branch's run_ga
# (design.py) needs a trained classifier's binding probability as its
# per-sequence fitness signal, which needs real ChIP-seq positives to train
# on. Without ChIP-seq there's no such classifier -- but a pretrained
# ChromBPNet model still lets us ask "how much does this position drive the
# model's accessibility prediction", via input x gradient (one backward
# pass, no extra forward passes -- unlike ISM). Averaging that over a
# candidate TF's best PWM-match window gives a fast, differentiable-free-at-
# use-time proxy for "is this TF plausibly bound here", standing in for the
# ChIP-seq-derived positive/negative label this branch doesn't have.
# ---------------------------------------------------------------------------


def gradient_importance_batch(seqs, cell_type: str, cfg) -> np.ndarray:
    """(N, INPUT_LEN) input x gradient importance, one row per sequence,
    from this cell type's pretrained ChromBPNet model. Positive values mean
    the observed base at that position is pushing predicted accessibility
    up; this is the fast standard "hypothetical importance" proxy (no
    per-position perturbation/ISM needed), computed with a single batched
    backward pass. Uses torch.autograd.grad rather than .backward() so
    this only computes the gradient wrt the input -- .backward() would
    also accumulate into the model's own parameter .grad tensors (never
    cleared, since nothing here is training the model), which is harmless
    numerically but wastes real memory and compute across every GA
    generation that calls this."""
    model = load_model(cell_type, cfg)
    x = torch.tensor(np.concatenate([_seq_to_onehot(s) for s in seqs], axis=0),
                      dtype=torch.float32, requires_grad=True)
    _, log_counts = model(x)
    grad_x, = torch.autograd.grad(log_counts.sum(), x)
    grad = grad_x.detach().numpy()      # (N, 4, L)
    onehot = x.detach().numpy()         # (N, 4, L)
    return (grad * onehot).sum(axis=1)  # (N, L) -- keep only the score at the observed base


def auto_detect_important_positions(seq: str, cell_type: str, cfg, top_fraction: float = 0.05) -> frozenset:
    """Auto-detected version of the manual lowercase-marking mechanism in
    design._parse_fixed_positions: instead of the user marking positions
    to protect by hand, rank every position by |gradient_importance_batch|
    (magnitude, not sign -- a position can matter by pushing the score
    either up or down) and return the indices of the top `top_fraction`
    as the positions to protect. Explicitly the "better version" flagged
    as optional/future-work in the original design brief -- reuses the
    same gradient-importance machinery gap 1 already built for TF-specific
    design, so it's cheap now rather than a separate feature. Still
    deliberately opt-in (see run_ga_chrombpnet's auto_fix_top_fraction),
    not the default -- manual marking stays the primary, predictable
    mechanism; this is a convenience on top of it, not a replacement."""
    importance = gradient_importance_batch([seq], cell_type, cfg)[0]
    n_fixed = max(1, int(round(len(seq) * top_fraction)))
    top_idx = np.argsort(-np.abs(importance))[:n_fixed]
    return frozenset(int(i) for i in top_idx)


def _motif_window(seq: str, tf: str, ppm_cache_path: str):
    """Best PWM match for tf in seq (either strand), mapped back to
    forward-strand coordinates so it indexes directly into
    gradient_importance_batch's output. Returns (start, length) or None if
    no recognizable match exists. Same strand-handling/clamping convention
    as design.py's _seed_from_sequence."""
    ppm = trim_low_information_flanks(load_ppm(tf, ppm_cache_path))
    m = len(ppm)
    L = len(seq)
    pos_fwd, v_fwd = best_match(seq, ppm)
    rc = reverse_complement(seq)
    pos_rev, v_rev = best_match(rc, ppm)
    if max(v_fwd, v_rev) <= 0:
        return None
    if v_fwd >= v_rev:
        start = pos_fwd - m // 2
    else:
        start = L - (pos_rev - m // 2 + m)
    start = max(0, min(start, L - m))
    return start, m


def tf_motif_importance_batch(seqs, tf: str, cell_type: str, cfg) -> np.ndarray:
    """Mean gradient-importance within tf's best PWM-match window, one
    value per sequence -- the ChIP-seq-free proxy described above. NaN
    where no recognizable motif match exists at all (the caller should
    penalize this, not treat it as a real low score: a missing motif isn't
    "weak binding", it's "not this TF's site")."""
    importance = gradient_importance_batch(seqs, cell_type, cfg)
    scores = np.full(len(seqs), np.nan)
    for i, seq in enumerate(seqs):
        window = _motif_window(seq, tf, cfg.jaspar_pfm_cache)
        if window is None:
            continue
        start, length = window
        scores[i] = importance[i, start:start + length].mean()
    return scores
