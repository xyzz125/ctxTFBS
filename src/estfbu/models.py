"""Wraps the paper's own DenseLSTM_classi architecture (never reimplemented
or modified -- imported directly from the cloned repo) with config-driven
hyperparameters instead of hardcoded ones."""
import sys


def _import_seq_regression_model(deeptfbu_repo: str):
    # the paper ships slightly different copies of SeqRegressionModel.py in
    # different subfolders (identical architecture, just duplicated) -- any
    # one of them works, this uses the one under 4_deconv_seq_into_TFBU
    module_dir = f"{deeptfbu_repo}/4_deconv_seq_into_TFBU"
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    import SeqRegressionModel
    return SeqRegressionModel


def build_model(cfg):
    """Build a fresh (untrained) DenseLSTM_classi with hyperparameters from cfg.model."""
    srm = _import_seq_regression_model(cfg.deeptfbu_repo)
    m = cfg.model
    return srm.DenseLSTM_classi(
        input_nc=4,
        growth_rate=m.growth_rate,
        block_config=tuple(m.block_config),
        num_init_features=m.num_init_features,
        bn_size=m.bn_size,
        drop_rate=m.drop_rate,
        input_length=m.seq_len,
    )
