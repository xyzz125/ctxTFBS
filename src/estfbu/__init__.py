"""esTFBU — reusable, parameterized version of the DeepTFBU paper's
"has ChIP-seq" branch (data prep -> context model training -> genetic
algorithm sequence design), with a TF blacklist gate up front.

Nothing in here modifies the paper's own code under data/DeepTFBU_repo/ --
this package imports the paper's model architecture and PPM matrices and
re-implements the pipeline logic around them with config-driven paths and
TF/cell-type parameters instead of hardcoded ones.
"""
from .config import load_config

__all__ = ["load_config"]
