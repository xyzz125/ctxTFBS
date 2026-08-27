"""Load and validate the pipeline's YAML config -- the single place that
knows about real filesystem paths, so nothing else in this package has a
hardcoded path in it."""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
QUICKSTART_CONFIG_PATH = REPO_ROOT / "config" / "quickstart_config.yaml"

# Config keys known to hold filesystem paths -- needed because a bare
# directory name with no '/' in it (e.g. work_dir: estfbu_workdir_quickstart)
# can't be told apart from a plain string value (e.g. species: human) by
# shape alone.
_PATH_KEYS = {
    "genome_fasta", "deeptfbu_repo", "work_dir", "hepg2_pretrained_weights_dir",
    "blacklist_file", "atac_bed", "h3k4me1_bed", "h3k4me3_bed", "chip_bed_dir",
    "nobias_h5", "gene_quantifications_tsv", "tf_to_ensembl_map", "tf_pfm_map",
    "jaspar_pfm_cache", "precomputed_genome_scan",
}


def _resolve_relative_paths(d, base: Path, key=None):
    """Resolves relative filesystem paths in the config against `base`
    (the repo root) -- lets config files like quickstart_config.yaml use
    paths such as 'sample_data/beds/HepG2_ATAC.bed' that work no matter
    what directory the pipeline is actually run from. A value is
    resolved if its key is a known path field (_PATH_KEYS) or, as a
    fallback, if it looks path-like (has a '/', isn't already absolute,
    isn't a URL)."""
    if isinstance(d, dict):
        return {k: _resolve_relative_paths(v, base, key=k) for k, v in d.items()}
    if isinstance(d, list):
        return [_resolve_relative_paths(v, base, key=key) for v in d]
    if isinstance(d, str) and not d.startswith("/") and "://" not in d:
        if key in _PATH_KEYS or "/" in d:
            return str((base / d).resolve())
    return d


def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


class Config(SimpleNamespace):
    """Dot-accessible config, e.g. cfg.model.seq_len, cfg.cell_types.HepG2.atac_bed."""

    def cell_type(self, name: str):
        cts = vars(self.cell_types)
        if name not in cts:
            raise ValueError(
                f"Unknown cell type '{name}'. Configured cell types: {list(cts.keys())}. "
                f"Add it to config/default_config.yaml under 'cell_types:' to support it."
            )
        return cts[name]

    def work_path(self, *parts) -> Path:
        """A FILE path under work_dir -- ensures the parent directory
        exists, but not the returned path itself."""
        p = Path(self.work_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def work_dir_path(self, *parts) -> Path:
        """A DIRECTORY path under work_dir -- ensures the returned path
        itself exists (creating it if needed)."""
        p = Path(self.work_dir).joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path=None) -> Config:
    """path=None uses config/default_config.yaml (full-scale, needs the
    external data set up per README.md's "Data setup" section) unless the
    ESTFBU_CONFIG environment variable is set, in which case that path is
    used instead -- lets test_scripts/ (which all call bare load_config())
    run against config/quickstart_config.yaml's bundled sample_data/
    without editing every call site: ESTFBU_CONFIG=config/quickstart_config.yaml
    python3 test_scripts/test_regression.py. An explicit path= argument
    always wins over the environment variable."""
    if path is None:
        path = os.environ.get("ESTFBU_CONFIG", DEFAULT_CONFIG_PATH)
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _resolve_relative_paths(raw, REPO_ROOT)
    ns = _to_namespace(raw)
    cfg = Config(**vars(ns))
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def seed_everything(seed: int):
    """Seeds every global RNG this pipeline touches. Most randomness here
    already goes through an explicitly-seeded np.random.RandomState
    instance (see design.py's run_ga/run_ga_chrombpnet), which doesn't
    need this -- the real gap this closes is PyTorch's GLOBAL RNG, which
    nothing else seeds: model weight initialization (models.build_model)
    and DataLoader(shuffle=True) batch order (train.train_context_model)
    both draw from it. Even with this, bit-exact reproduction across
    different machines/thread counts isn't guaranteed (CPU floating-point
    summation isn't associative) -- see EXPECTED_OUTPUTS.md for real
    measured run-to-run variance rather than a claimed exact value."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
