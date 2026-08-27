"""RNA-seq expression lookup for TFs in HepG2/K562 -- the piece needed to
answer "is this TF actually expressed here" rather than just "does its
motif match here" (which is all motif_hits.py alone can say). This is the
color dimension for the bubble/dot plot idea discussed early in the
project (size=motif/ChromBPNet score, color=expression level).

Data: ENCODE polyA+ RNA-seq gene quantifications, GRCh38, standard
untreated whole-cell experiments (same accession-selection logic as the
paper's own ChIP-seq/ATAC-seq choices -- picked the plain "RNA-seq on
human {cell_type}" experiment, not a treatment/fractionation variant):
  HepG2: ENCSR329MHM (file ENCFF640ZBJ)
  K562:  ENCSR000EYO (file ENCFF179CNW)
TF symbol -> Ensembl gene ID mapping via mygene.info, cached in
data/processed/tf_to_ensembl.json (built once, see the fetch script this
module's docstring references -- not re-fetched at runtime).
"""
import json

import pandas as pd

_expr_cache = {}
_mapping_cache = None


def _load_mapping(cfg):
    global _mapping_cache
    if _mapping_cache is None:
        with open(cfg.tf_to_ensembl_map) as f:
            _mapping_cache = json.load(f)
    return _mapping_cache


def load_expression(cell_type: str, cfg) -> pd.DataFrame:
    """Returns a DataFrame indexed by base Ensembl gene ID (version
    stripped, e.g. 'ENSG00000179348' not 'ENSG00000179348.11') with a
    'TPM' column, for the configured RNA-seq quantification file."""
    if cell_type in _expr_cache:
        return _expr_cache[cell_type]

    rna = getattr(cfg, "rna_seq", None)
    if rna is None or not hasattr(rna, cell_type):
        raise ValueError(
            f"No rna_seq.{cell_type} entry in config. Add the path to that cell type's "
            f"ENCODE gene quantification TSV."
        )
    path = getattr(rna, cell_type).gene_quantifications_tsv
    df = pd.read_csv(path, sep="\t")
    df["gene_id_base"] = df["gene_id"].str.split(".").str[0]
    df = df.set_index("gene_id_base")
    _expr_cache[cell_type] = df
    return df


def get_tpm(tf: str, cell_type: str, cfg) -> float:
    """TPM for a TF's gene in a given cell type. Returns np.nan if the TF
    couldn't be mapped to an Ensembl gene ID, or (confirmed for 6/198 TFs
    in tf_to_ensembl.json -- RXRB, IRF9, MEF2D, PBX2, TFE3, ZNF707, via
    Ensembl's own REST API) the gene's only current Ensembl model sits on
    an alternate-haplotype/patch scaffold rather than the primary GRCh38
    assembly, which ENCODE's standard RNA-seq quantification doesn't
    cover -- the mapping is correct in these cases, there's just no
    primary-assembly gene ID to map to instead. Not a bug to "fix" by
    re-mapping; see EXPECTED_OUTPUTS.md's scenario 6 for the investigation."""
    mapping = _load_mapping(cfg)
    gene_id = mapping.get(tf)
    if gene_id is None:
        return float("nan")
    df = load_expression(cell_type, cfg)
    if gene_id not in df.index:
        return float("nan")
    return float(df.loc[gene_id, "TPM"])


def annotate_tpm(regions_df, tf_column: str, cell_types, cfg):
    """regions_df: DataFrame with a column of TF names (tf_column, e.g.
    from motif_hits output parsed back out). Adds one '{cell_type}_tpm'
    column per cell type in cell_types."""
    regions_df = regions_df.copy()
    for ct in cell_types:
        regions_df[f"{ct}_tpm"] = regions_df[tf_column].apply(lambda tf: get_tpm(tf, ct, cfg))
    return regions_df
