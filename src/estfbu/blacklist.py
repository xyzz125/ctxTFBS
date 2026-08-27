"""User-editable TF blacklist. Plain text, one TF per line, # for comments."""
from pathlib import Path


class BlacklistedError(Exception):
    pass


def load_blacklist(path) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line)
    return names


def check(tf: str, blacklist_path) -> None:
    """Raise BlacklistedError if tf is blacklisted; otherwise return silently."""
    blacklisted = load_blacklist(blacklist_path)
    if tf in blacklisted:
        raise BlacklistedError(
            f"'{tf}' is in the blacklist ({blacklist_path}). "
            f"Edit that file to remove it if this TF should be allowed."
        )


def available_chipseq_tfs(cell_types, cfg):
    """TF names that actually have a ChIP-seq bed file for every one of
    the given cell types (has-ChIP-seq branch needs one per cell type
    involved in the comparison -- e.g. HNF4A only has a bundled bed for
    HepG2, not K562, so a HepG2-vs-K562 request must exclude it)."""
    per_cell_type_sets = []
    for ct in cell_types:
        chip_dir = Path(cfg.cell_type(ct).chip_bed_dir)
        prefix, suffix = f"{ct}_ChIP_", ".bed"
        tfs = {p.name[len(prefix):-len(suffix)] for p in chip_dir.glob(f"{prefix}*{suffix}")}
        per_cell_type_sets.append(tfs)
    return sorted(set.intersection(*per_cell_type_sets)) if per_cell_type_sets else []


def exclude_blacklisted(tfs, cfg):
    """Drops blacklisted TFs from a candidate list BEFORE it's ever shown
    as an option -- the blacklist gate belongs at the front of TF
    selection, not as a check run reactively after one is already picked."""
    blocked = load_blacklist(cfg.blacklist_file)
    return [tf for tf in tfs if tf not in blocked]
