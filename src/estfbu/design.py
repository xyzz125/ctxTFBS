"""Parameterized version of the paper's genetic-algorithm sequence design
(0_HepG2_specific_optimize_masked.py and friends), generalized to:
  - target='activity': maximize a single cell type's context score
  - target='specificity': maximize (comparative_A_vs_B + context_A - context_B)
for any TF / cell-type pair, instead of hardcoded GATA2/HepG2/K562 + CUDA.
"""
import math

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from . import data_prep, train
from .motif_scoring import best_match, load_ppm, reverse_complement, trim_low_information_flanks


def _parse_fixed_positions(seq: str):
    """The whiteboard's manual "固定部分序列" (fix part of a starting
    sequence) case: lowercase letters mark positions to protect from
    mutation/recombination for the whole GA run; uppercase are free to
    optimize. An all-uppercase input (the common case) returns an empty
    set, so this is fully backward compatible. Deliberately manual, not
    auto-detected -- per the design brief, auto-detecting "important"
    regions is future work, not needed for a first version."""
    fixed = frozenset(i for i, c in enumerate(seq) if c.islower())
    return seq.upper(), fixed


def _seed_from_sequence(tf: str, starting_sequence: str, cfg, pool_size: int):
    """Seed the GA from a user-given sequence instead of real observed
    windows -- the whiteboard's "给定初始序列" (given a starting sequence)
    case. Locates the TF's core motif within the given sequence (best PPM
    match, either strand) and masks it the same way data_prep does, so the
    GA still respects the "optimize the context, not the core motif"
    invariant everywhere else in this pipeline. Raises if the sequence
    doesn't contain a recognizable match for the TF's motif at all --
    silently masking the wrong region would be worse than failing loudly.

    Also honors any lowercase-marked fixed positions (see
    _parse_fixed_positions) elsewhere in the sequence -- both mechanisms
    protect positions from mutation, they just do it for different reasons
    (one because it's the TF's own binding site, one because the user said
    so), and can coexist. Returns (seed_onehot, protected_indices) so the
    caller can keep honoring the fixed positions for the rest of the run,
    not just this initial seeding step.
    """
    seq_len = cfg.model.seq_len
    if len(starting_sequence) != seq_len:
        raise ValueError(f"starting_sequence must be exactly {seq_len}bp, got {len(starting_sequence)}")
    starting_sequence, fixed_fwd = _parse_fixed_positions(starting_sequence)

    ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
    m = len(ppm)
    pos_fwd, v_fwd = best_match(starting_sequence, ppm)
    rc = reverse_complement(starting_sequence)
    pos_rev, v_rev = best_match(rc, ppm)

    if max(v_fwd, v_rev) <= 0:
        raise ValueError(
            f"No recognizable '{tf}' motif match found anywhere in the given starting "
            f"sequence -- refusing to guess where to mask. Check the sequence is correct "
            f"and long enough to contain a real binding site."
        )

    use_rc = v_rev > v_fwd
    base_seq = rc if use_rc else starting_sequence
    # user-marked positions were given in forward-strand coordinates -- flip
    # them to match base_seq if the motif matched better on the reverse strand
    fixed = frozenset(seq_len - 1 - i for i in fixed_fwd) if use_rc else fixed_fwd

    core_start = (pos_rev if use_rc else pos_fwd) - m // 2
    core_start = max(0, min(core_start, seq_len - m))
    masked = base_seq[:core_start] + "N" * m + base_seq[core_start + m:]
    # anything inside the auto-masked core motif is already protected by
    # the "N" convention (and gets the real consensus spliced back in
    # during postprocessing either way), so drop it from the explicit set
    fixed = frozenset(i for i in fixed if not (core_start <= i < core_start + m))

    # seed the pool with the masked starting sequence plus light mutational
    # variants of it, so the GA has room to explore around the user's
    # anchor point rather than being stuck on one exact sequence
    rng = np.random.RandomState(cfg.training.random_seed)
    pool = [masked]
    while len(pool) < pool_size:
        pool.append(_mutate(masked, rng, fixed))
    return _seq_to_onehot(pool[:pool_size], seq_len), fixed


def _seed_population(tf: str, cell_types, cfg, max_pool_size: int):
    """Seed the GA's starting population from real observed masked windows
    (last 10% split, same convention as the paper -- avoids ever "designing"
    starting from nothing, which the paper doesn't do either)."""
    if len(cell_types) == 2:
        d = cfg.work_dir_path("data_prep", f"{cell_types[0]}_vs_{cell_types[1]}")
        h5_path = d / f"step0_{tf}.h5"
        if not h5_path.exists():
            train._comparative_h5(tf, cell_types[0], cell_types[1], cfg)
        with h5py.File(h5_path) as f:
            pos = f[f"pos_{tf}"][:]
            neg = f[f"neg_{tf}"][:]
        seqs = np.concatenate([pos, neg])
    else:
        h5_path = data_prep.prepare(tf, cell_types[0], cfg)
        with h5py.File(h5_path) as f:
            pos = f[f"pos_{tf}"][:]
        seqs = pos

    rng = np.random.RandomState(cfg.training.random_seed)
    seqs = seqs[rng.permutation(len(seqs))]
    n_test = max(1, int(len(seqs) * 0.1))
    seed = seqs[-n_test:]
    if len(seed) > max_pool_size:
        seed = seed[:max_pool_size]
    return seed


_MAPPING = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}


def _onehot_to_seq(vec) -> str:
    out = []
    for row in vec:
        out.append("N" if row.sum() == 0 else _MAPPING[int(np.argmax(row))])
    return "".join(out)


def _seq_to_onehot(seqs, length) -> np.ndarray:
    """length is required (not defaulted) -- this used to default to 168
    and both call sites omitted it, so cfg.model.seq_len was silently
    ignored: a non-168 seq_len either crashed with an IndexError (length
    > 168) or got zero-padded with no error at all (length < 168),
    producing a corrupted-looking-plausible encoding. Raising ValueError
    (not a bare `assert`, which `python -O` strips entirely -- exactly
    the wrong statement for a check whose whole job is turning silent
    corruption loud) on any length mismatch turns that second, silent
    case into a loud one instead."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for s in seqs:
        if len(s) != length:
            raise ValueError(f"expected all sequences to be {length}bp, got one of {len(s)}bp")
    arr = np.zeros((len(seqs), length, 4))
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            if ch in idx:
                arr[i, j, idx[ch]] = 1
    return arr


def _mutate(seq: str, rng, protected: frozenset = frozenset()) -> str:
    """protected: indices to never touch, on top of the always-protected
    'N' convention -- see _parse_fixed_positions. Bails out unchanged if
    every position is protected/N rather than looping forever (a
    pathological case: a starting sequence with nothing left to mutate)."""
    z = list(seq)
    attempts = 0
    p = rng.randint(0, len(z))
    while z[p] == "N" or p in protected:
        p = rng.randint(0, len(z))
        attempts += 1
        if attempts > 10000:
            return seq
    z[p] = "ACGT"[rng.randint(0, 4)]
    return "".join(z)


def _recombine(seq: str, pool, rng, protected: frozenset = frozenset()) -> str:
    z = list(seq)
    partner = list(pool[rng.randint(0, len(pool))])
    mask = rng.randint(0, 2, size=len(z))
    for i in range(len(z)):
        if mask[i] == 1 and i not in protected:
            z[i] = partner[i]
    return "".join(z)


def _select_parents(n_new, n_elite, pool_size, rng):
    from_elite = min(n_elite, n_new // 2)
    from_rest = min(pool_size - n_elite, n_new - from_elite)
    idx_elite = rng.choice(n_elite, size=from_elite, replace=False) if from_elite else np.array([], dtype=int)
    rest_pool = np.arange(n_elite, pool_size)
    idx_rest = rng.choice(rest_pool, size=from_rest, replace=False) if from_rest else np.array([], dtype=int)
    return np.concatenate([idx_elite, idx_rest]).astype(int)


def _score_batch(model, onehot):
    with torch.no_grad():
        p = F.softmax(model(torch.tensor(onehot, dtype=torch.float32)), dim=1).numpy()[:, 1]
    return p


def run_ga(tf: str, cell_types, target: str, cfg, progress_every: int = 50, starting_sequence: str = None):
    """cell_types: [A] for target='activity', or [A, B] for
    target='specificity' (design sequences specific to A over B).
    starting_sequence: optional user-given seq_len-bp sequence to seed and
    anchor the search around (the whiteboard's "给定初始序列" case) instead
    of seeding from real observed ChIP-seq-derived windows -- use this when
    optimizing a specific region of interest rather than designing de novo.
    Lowercase letters in it mark positions to keep fixed for the whole run
    (see _parse_fixed_positions) -- e.g. a known-important element you don't
    want the GA mutating away, on top of whatever TF motif gets auto-masked.
    Returns (final_sequences: list[str], final_scores: np.ndarray)."""
    torch.set_num_threads(cfg.training.cpu_threads)
    ga = cfg.genetic_algorithm
    seq_len = cfg.model.seq_len
    rng = np.random.RandomState(cfg.training.random_seed)

    if target == "specificity":
        if len(cell_types) != 2:
            raise ValueError("target='specificity' requires exactly 2 cell_types")
        a, b = cell_types
        model_a = train.get_or_train_model(tf, a, cfg)
        model_b = train.get_or_train_model(tf, b, cfg)
        model_cmp = train.get_or_train_model(tf, a, cfg, comparative_with=b)

        def score_fn(onehot):
            return (_score_batch(model_cmp, onehot) + _score_batch(model_a, onehot)
                    - _score_batch(model_b, onehot))
    elif target == "activity":
        a = cell_types[0]
        model_a = train.get_or_train_model(tf, a, cfg)

        def score_fn(onehot):
            return _score_batch(model_a, onehot)
    else:
        raise ValueError("target must be 'activity' or 'specificity'")

    protected = frozenset()
    if starting_sequence:
        seed_onehot, protected = _seed_from_sequence(tf, starting_sequence, cfg, ga.max_pool_size)
    else:
        seed_onehot = _seed_population(tf, cell_types, cfg, ga.max_pool_size)
    target_pool_size = len(seed_onehot)  # cap to (re)grow toward each iteration
    seqs = np.array([_onehot_to_seq(v) for v in seed_onehot])
    scores = score_fn(seed_onehot)

    # Archive of every unique sequence seen across ALL iterations, not just
    # the final converged population. This matters: with no explicit GC
    # penalty in the fitness function, the final population can (and does)
    # drift GC content as it converges toward the highest-scoring region of
    # sequence space, potentially failing postprocess's GC filter entirely
    # if only the last iteration's pool is considered. The paper's own
    # scripts sidestep this by dumping every iteration's population to disk
    # and scanning backward through all of them during post-processing;
    # this in-memory archive is the equivalent for this refactor.
    archive = {s: sc for s, sc in zip(seqs, scores)}

    for it in range(ga.n_iterations):
        # size _select_parents off the CURRENT population, not the original
        # target -- if dedup ever shrinks the population below
        # target_pool_size (common when seeding from a single
        # starting_sequence, since repeated single-point mutations of the
        # same anchor collide often), _select_parents must not be handed a
        # stale, too-large size or it indexes past the end of the actual
        # array. The final truncation below still caps toward
        # target_pool_size so the population can grow back over time as
        # more unique candidates are discovered.
        pool_size = len(seqs)
        order = np.argsort(-scores)
        seqs, scores = seqs[order], scores[order]

        n_elite = math.ceil(pool_size * ga.pct_elite)
        n_new = math.ceil(pool_size * ga.pct_new)
        parent_idx = _select_parents(n_new, n_elite, pool_size, rng)

        new_seqs = []
        for i in parent_idx:
            if rng.randint(0, 2) == 0:
                new_seqs.append(_mutate(seqs[i], rng, protected))
            else:
                new_seqs.append(_recombine(seqs[i], seqs, rng, protected))
        new_onehot = _seq_to_onehot(new_seqs, seq_len)
        new_scores = score_fn(new_onehot)

        for s, sc in zip(new_seqs, new_scores):
            if s not in archive:
                archive[s] = sc

        all_seqs = np.concatenate([seqs, new_seqs])
        all_scores = np.concatenate([scores, new_scores])
        _, unique_idx = np.unique(all_seqs, return_index=True)
        all_seqs, all_scores = all_seqs[unique_idx], all_scores[unique_idx]

        order = np.argsort(-all_scores)[:target_pool_size]
        seqs, scores = all_seqs[order], all_scores[order]

        if progress_every and (it + 1) % progress_every == 0:
            print(f"  GA iter {it+1}/{ga.n_iterations}, best score={scores[0]:.4f}, "
                  f"archive size={len(archive)}")

    archive_seqs = np.array(list(archive.keys()))
    archive_scores = np.array(list(archive.values()))
    order = np.argsort(-archive_scores)
    return list(archive_seqs[order]), archive_scores[order]


# ---------------------------------------------------------------------------
# ChromBPNet-based design (no-ChIP-seq branch): the has-ChIP-seq run_ga above
# optimizes a TF-specific 168bp "context around a masked core motif", using
# a classifier trained on real ChIP-seq positives/negatives. This variant
# optimizes a full 2114bp sequence against a pretrained ChromBPNet model
# instead -- no masking (there's no fixed "core" position to protect), and
# by default no TF concept at all (ChromBPNet isn't per-TF, so the whole
# sequence is free to mutate against raw predicted accessibility). Passing
# tf= switches the fitness function to chrombpnet_scoring's gradient-based
# motif-importance proxy instead, making a TF-specific design possible even
# without ChIP-seq data -- see run_ga_chrombpnet's docstring.
# ---------------------------------------------------------------------------

CHROMBPNET_INPUT_LEN = 2114


def _seed_from_atac_peaks(cell_type: str, cfg, max_pool_size: int):
    """Seed from real ATAC peak regions (2114bp windows), same spirit as
    _seed_population -- start from real observed accessible sequence, not
    from nothing."""
    from pyfaidx import Fasta
    genome = Fasta(cfg.genome_fasta)
    ct = cfg.cell_type(cell_type)
    half = CHROMBPNET_INPUT_LEN // 2

    seqs = []
    with open(ct.atac_bed) as f:
        for line in f:
            parts = line.split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            center = (start + end) // 2
            seq = genome[chrom][center - half:center + half].seq.upper()
            if len(seq) == CHROMBPNET_INPUT_LEN and "N" not in seq:
                seqs.append(seq)
            if len(seqs) >= max_pool_size:
                break
    return seqs


def run_ga_chrombpnet(cell_types, target: str, cfg, progress_every: int = 50,
                       starting_sequence: str = None, seed_pool_size: int = None,
                       tf: str = None, auto_fix_top_fraction: float = None):
    """ChromBPNet-driven design. cell_types: [A] for target='activity'
    (maximize A's predicted accessibility), or [A, B] for
    target='specificity' (maximize A's score minus B's score, i.e. design
    sequences predicted accessible in A but not B).

    tf: optional. Without it, no TF-specific masking concept applies here
    (ChromBPNet isn't per-TF) -- the whole sequence is free to mutate and
    the fitness is raw predicted accessibility, same as always. With it,
    fitness instead becomes chrombpnet_scoring.tf_motif_importance_batch:
    mean input-x-gradient importance within tf's best PWM-match window,
    the ChIP-seq-free proxy for "is this TF plausibly bound here" (see that
    module's docstring). A candidate with no recognizable tf motif match at
    all scores -1e9 (heavily penalized, not just "low") so the GA is
    pushed toward keeping/creating a real match rather than drifting away
    from the TF entirely.

    Lowercase letters in starting_sequence mark positions to keep fixed for
    the whole run (see _parse_fixed_positions) -- unlike the has-ChIP-seq
    branch, there's no auto-masked TF motif here, so this is the only
    always-available way to protect part of a starting sequence in this
    branch. auto_fix_top_fraction (optional): additionally auto-detect and
    protect the top fraction of positions by |gradient importance| in
    cell_types[0] (see chrombpnet_scoring.auto_detect_important_positions)
    -- unioned with any manually-marked positions, not a replacement for
    them. Deliberately opt-in, not the default.

    Returns (final_sequences: list[str], final_scores: np.ndarray), same
    archive-across-all-iterations approach as run_ga, for the same reason
    (avoid the final-population-GC-drift problem)."""
    from . import chrombpnet_scoring as cbp

    torch.set_num_threads(cfg.training.cpu_threads)
    ga = cfg.genetic_algorithm
    rng = np.random.RandomState(cfg.training.random_seed)
    seed_pool_size = seed_pool_size or ga.max_pool_size

    if target == "specificity" and len(cell_types) != 2:
        raise ValueError("target='specificity' requires exactly 2 cell_types")

    if tf:
        def score_fn(seqs):
            imp_a = cbp.tf_motif_importance_batch(seqs, tf, cell_types[0], cfg)
            if target == "specificity":
                imp_b = cbp.tf_motif_importance_batch(seqs, tf, cell_types[1], cfg)
                imp_a = imp_a - imp_b
            return np.nan_to_num(imp_a, nan=-1e9)
    else:
        model_a = cbp.load_model(cell_types[0], cfg)
        if target == "specificity":
            model_b = cbp.load_model(cell_types[1], cfg)

        def score_fn(seqs):
            x = torch.tensor(np.concatenate([cbp._seq_to_onehot(s) for s in seqs], axis=0))
            with torch.no_grad():
                _, log_counts_a = model_a(x)
            scores = log_counts_a.numpy().flatten()
            if target == "specificity":
                with torch.no_grad():
                    _, log_counts_b = model_b(x)
                scores = scores - log_counts_b.numpy().flatten()
            return scores

    protected = frozenset()
    if starting_sequence:
        if len(starting_sequence) != CHROMBPNET_INPUT_LEN:
            raise ValueError(f"starting_sequence must be exactly {CHROMBPNET_INPUT_LEN}bp for "
                              f"the ChromBPNet branch, got {len(starting_sequence)}")
        starting_sequence, protected = _parse_fixed_positions(starting_sequence)
        if auto_fix_top_fraction:
            protected = protected | cbp.auto_detect_important_positions(
                starting_sequence, cell_types[0], cfg, top_fraction=auto_fix_top_fraction)
        seqs = np.array([starting_sequence] +
                         [_mutate(starting_sequence, rng, protected) for _ in range(seed_pool_size - 1)])
    else:
        seed_seqs = _seed_from_atac_peaks(cell_types[0], cfg, seed_pool_size)
        seqs = np.array(seed_seqs)

    target_pool_size = len(seqs)
    scores = score_fn(seqs)
    archive = {s: sc for s, sc in zip(seqs, scores)}

    for it in range(ga.n_iterations):
        pool_size = len(seqs)
        order = np.argsort(-scores)
        seqs, scores = seqs[order], scores[order]

        n_elite = math.ceil(pool_size * ga.pct_elite)
        n_new = math.ceil(pool_size * ga.pct_new)
        parent_idx = _select_parents(n_new, n_elite, pool_size, rng)

        new_seqs = []
        for i in parent_idx:
            if rng.randint(0, 2) == 0:
                new_seqs.append(_mutate(seqs[i], rng, protected))
            else:
                new_seqs.append(_recombine(seqs[i], seqs, rng, protected))
        new_scores = score_fn(new_seqs)

        for s, sc in zip(new_seqs, new_scores):
            if s not in archive:
                archive[s] = sc

        all_seqs = np.concatenate([seqs, new_seqs])
        all_scores = np.concatenate([scores, new_scores])
        _, unique_idx = np.unique(all_seqs, return_index=True)
        all_seqs, all_scores = all_seqs[unique_idx], all_scores[unique_idx]

        order = np.argsort(-all_scores)[:target_pool_size]
        seqs, scores = all_seqs[order], all_scores[order]

        if progress_every and (it + 1) % progress_every == 0:
            print(f"  [chrombpnet GA] iter {it+1}/{ga.n_iterations}, best score={scores[0]:.4f}, "
                  f"archive size={len(archive)}")

    archive_seqs = np.array(list(archive.keys()))
    archive_scores = np.array(list(archive.values()))
    order = np.argsort(-archive_scores)
    return list(archive_seqs[order]), archive_scores[order]
