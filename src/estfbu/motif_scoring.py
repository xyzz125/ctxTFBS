"""Shared PPM-scanning utilities used throughout data prep and design.

Note on PFM source: the paper's own scripts hardcode each TF's raw JASPAR
count matrix inline (e.g. GATA2's matrix pasted directly into 5+ different
files). That doesn't scale to "any TF" or stay in sync if JASPAR updates a
matrix. Instead this module loads PFMs from data/processed/jaspar_pfms.json
(built once via the JASPAR REST API, see archive/scripts/02_fetch_jaspar_pfms.py),
which already covers all 198 TFs used in the paper -- one source of truth
instead of N hardcoded copies.
"""
import json

import numpy as np
from numba import jit

BASES = "ACGT"
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

_pfm_cache = {}


def _load_pfms(jaspar_json_path: str) -> dict:
    """Cached parse of jaspar_pfms.json, keyed by path -- load_ppm gets
    called once per TF per region (e.g. annotate_regions calls it 198
    times per region, once per candidate TF), and re-opening/re-parsing
    the whole file every time was ~4000 full JSON parses for a 20-region
    annotation. The file is static for the lifetime of a run (nothing
    writes to jaspar_pfms.json while the pipeline runs), so this is safe."""
    if jaspar_json_path not in _pfm_cache:
        with open(jaspar_json_path) as f:
            _pfm_cache[jaspar_json_path] = json.load(f)
    return _pfm_cache[jaspar_json_path]


def load_ppm(tf: str, jaspar_json_path: str) -> np.ndarray:
    """Return an (L, 5) PPM matrix in [A, C, G, T, N=0.25] column order,
    matching the format the paper's own cal_max_pos()/res_dic convention
    expects (N always scores 0.25, i.e. never the best or worst base)."""
    pfms = _load_pfms(jaspar_json_path)
    if tf not in pfms:
        raise ValueError(f"No cached JASPAR PFM for '{tf}'. Add it to data/processed/jaspar_pfms.json first.")
    counts = np.array([pfms[tf]["pfm"][b] for b in BASES], dtype=np.float64).T  # (L, 4)
    ppm = counts / counts.sum(axis=1, keepdims=True)
    ppm = np.concatenate([ppm, np.full((len(ppm), 1), 0.25)], axis=1)  # (L, 5)
    return ppm


def trim_low_information_flanks(ppm: np.ndarray, info_content_thresh: float = 0.3) -> np.ndarray:
    """Drop low-information-content positions off both ends of the PPM,
    same rule the paper uses to define the 'core' motif length."""
    info = np.zeros(len(ppm))
    for i in range(len(ppm)):
        p = ppm[i, :4]
        p = p[p > 0]
        info[i] = 2 + np.sum(p * np.log2(p))
    to_del = []
    for i in range(len(info)):
        if info[i] < info_content_thresh:
            to_del.append(i)
        else:
            break
    for i in range(len(info)):
        j = len(info) - 1 - i
        if info[j] < info_content_thresh:
            to_del.append(j)
        else:
            break
    return np.delete(ppm, to_del, axis=0)


def consensus_sequence(ppm: np.ndarray) -> str:
    return "".join(BASES[int(np.argmax(row[:4]))] for row in ppm)


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[b] for b in reversed(seq))


@jit(nopython=True)
def _scan_max(seq, ppm):
    res = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    L = len(ppm)
    n = len(seq) + 1 - L
    vals = np.zeros(n)
    for i in range(n):
        v = 1.0
        for k in range(L):
            v *= ppm[k][res[seq[i + k]]]
        vals[i] = v
    return vals


def best_match(seq: str, ppm: np.ndarray, pos_orig: int = None):
    """Scan seq with ppm, return (best_position(s), best_value).
    If pos_orig given and multiple positions tie for best, pick the one
    closest to pos_orig (matches the paper's tie-break rule for negative
    samples, where the 'true' position is ambiguous)."""
    vals = _scan_max(seq, ppm)
    best_val = np.max(vals)
    positions = np.argwhere(vals == best_val).flatten() + len(ppm) // 2
    if pos_orig is not None and len(positions) > 1:
        best_pos = positions[np.argmin(np.abs(positions - pos_orig))]
    else:
        best_pos = positions[0]
    return int(best_pos), float(best_val)
