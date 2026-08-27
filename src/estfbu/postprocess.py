"""Parameterized version of the paper's post-processing filters
(1_HepG2_specific_check_and_selec_gc10_aim.py / 2_..._edit50_check_...py):
drop sequences where the full TF consensus motif leaked into the context,
match GC content to the training distribution, remove restriction sites,
and deduplicate by edit distance to keep the final set genuinely diverse.

dedup_by_edit_distance() is the TF-agnostic core (restriction-site removal +
edit-distance dedup + top-N cap) shared by both the chipseq branch
(filter_and_dedup, which also does the TF-motif-specific GC/motif-leak
checks on top) and the chrombpnet branch (which has no TF-specific core
motif to check, so only needs this generic part).
"""
import Levenshtein

from .motif_scoring import consensus_sequence, load_ppm, reverse_complement, trim_low_information_flanks


def dedup_by_edit_distance(seqs, scores, cfg, check_restriction_sites: bool = True):
    """TF-agnostic filtering: drop sequences containing a restriction site
    (only meaningful if the output might get cloned in a wet lab -- skip
    with check_restriction_sites=False for e.g. purely computational
    ChromBPNet-design candidates where that's not the point), deduplicate
    by edit distance so the final set is genuinely diverse rather than
    near-identical variants of the same top hit, cap at
    cfg.postprocess.n_final_sequences. seqs/scores: aligned, higher score
    = better; doesn't require pre-sorting.

    The edit-distance threshold is cfg.postprocess.edit_distance_fraction
    of the actual sequence length, not one fixed absolute number -- this
    function is shared by the has-ChIP-seq branch's 168bp context
    sequences and the no-ChIP-seq branch's 2114bp ChromBPNet sequences,
    and a single absolute threshold means two very different things at
    those two scales (a real ~30% diversity requirement at 168bp is a
    trivial ~2% one at 2114bp, where different genomic loci are already
    hundreds of edits apart regardless)."""
    pp = cfg.postprocess
    if not seqs:
        return []
    threshold = round(len(seqs[0]) * pp.edit_distance_fraction)
    order = sorted(range(len(seqs)), key=lambda i: -scores[i])

    selected = []
    for i in order:
        seq = seqs[i]
        if check_restriction_sites and (any(site in seq for site in pp.restriction_sites) or "N" in seq):
            continue
        if any(Levenshtein.distance(seq, prev) < threshold for prev in selected):
            continue
        selected.append(seq)
        if len(selected) >= pp.n_final_sequences:
            break
    return selected


def filter_and_dedup(tf: str, seqs, scores, cfg, target_gc: float = 0.5):
    """seqs: list[str] of full (unmasked, or masked-then-reconstructed)
    168bp sequences, aligned with scores (higher = better, already sorted
    descending is fine but not required -- this re-sorts)."""
    pp = cfg.postprocess
    ppm = trim_low_information_flanks(load_ppm(tf, cfg.jaspar_pfm_cache))
    motif = consensus_sequence(ppm)
    motif_rc = reverse_complement(motif)
    m = len(motif)
    half = cfg.model.seq_len // 2
    core_start = half - m // 2

    order = sorted(range(len(seqs)), key=lambda i: -scores[i])

    # TF-specific pass: unmask (N -> real consensus motif) first, then run
    # the motif-leak and GC-content checks against the actual final
    # sequence -- both checks used to run on the still-masked seq instead:
    # the GC check was measuring the N-placeholder's composition rather
    # than the real spliced-in motif's (a ~2% shift for a 7bp GATA2-style
    # motif in a 168bp sequence, small but wrong relative to the exact
    # target_gc this filters on), and the leak check only ever tested the
    # forward-strand motif string, so a reverse-complement match sitting
    # in the context (equally a real "motif leaked into context" case)
    # passed undetected.
    candidates, candidate_scores = [], []
    for i in order:
        seq = seqs[i]
        full_seq = seq[:core_start] + motif + seq[core_start + m:] if "N" * m in seq else seq
        context = full_seq[:core_start] + full_seq[core_start + m:]  # everything but the core
        if motif in context or motif_rc in context:
            continue  # consensus motif (either strand) leaked into context -- not a real context effect
        gc = sum(c in "GC" for c in full_seq) / sum(c in "ACGT" for c in full_seq)
        if abs(gc - target_gc) > pp.gc_content_tolerance:
            continue
        candidates.append(full_seq)
        candidate_scores.append(scores[i])

    # generic pass: restriction sites + edit-distance dedup + cap
    return dedup_by_edit_distance(candidates, candidate_scores, cfg, check_restriction_sites=True)
