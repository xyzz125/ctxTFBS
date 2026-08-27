"""Standardized step reporting shared by both the interactive script
(run_pipeline.py) and the CLI (cli.py) -- previously only implemented in
run_pipeline.py, which meant the CLI could produce a ranked TF list or a
designed-sequence comparison but never the bubble plot or GC-comparison
plot that go with them, and every fix here had to be applied twice."""
import json
import time
from pathlib import Path

import numpy as np

from .expression import get_tpm


def write_step_manifest(cfg, step: str, scenario: str, output_path, output_format: str,
                         n_records: int, seed: int, elapsed_sec: float):
    """Writes a small standardized JSON summary next to a step's real
    output file -- same shape (step, scenario, output_path, output_format,
    n_records, seed, elapsed_sec, timestamp) no matter which pathway
    produced it, even though the underlying artifact (FASTA vs CSV vs h5)
    differs. This is what makes results comparable/scriptable across
    scenarios -- see EXPECTED_OUTPUTS.md."""
    output_path = Path(output_path)
    manifest = {
        "step": step,
        "scenario": scenario,
        "output_path": str(output_path),
        "output_format": output_format,
        "n_records": n_records,
        "seed": seed,
        "elapsed_sec": round(elapsed_sec, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest: {manifest_path}")
    return manifest_path


def write_bubble_plot(tf_names, scores_a, scores_b, cell_types, cfg, out_path, title):
    """Renders a two-column bubble plot: one column per cell type, bubble
    size = that TF's score (accessibility or model probability, whatever
    the caller passes in), color = RNA-seq expression (log10 TPM+1, via
    get_tpm). Bubble radii are computed from the actual rendered row
    spacing (not a fixed guess) so rows never overlap regardless of how
    many TFs are plotted -- see the max_radius_points/min_radius_points
    derivation below.

    tf_names/scores_a/scores_b: aligned, same length, one entry per TF.
    Used by both branches' 'no TF specified' screening modes so a ranked
    list always comes with a visual, not just a CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n = len(tf_names)
    y = np.arange(n)[::-1]
    tpm_a = np.array([get_tpm(tf, cell_types[0], cfg) for tf in tf_names])
    tpm_b = np.array([get_tpm(tf, cell_types[1], cfg) for tf in tf_names])
    log_tpm = lambda t: np.log10(np.nan_to_num(t, nan=0.0) + 1)
    all_log_tpm = np.concatenate([log_tpm(tpm_a), log_tpm(tpm_b)])
    norm = mcolors.Normalize(vmin=all_log_tpm.min(), vmax=max(all_log_tpm.max(), all_log_tpm.min() + 1e-6))
    cmap = mcolors.LinearSegmentedColormap.from_list("blues", ["#E6F1FB", "#0C447C"])

    scores_a, scores_b = np.array(scores_a, dtype=float), np.array(scores_b, dtype=float)
    score_min = min(np.nanmin(scores_a), np.nanmin(scores_b))
    score_max = max(np.nanmax(scores_a), np.nanmax(scores_b))
    score_range = max(score_max - score_min, 1e-9)

    fig_height = max(3.0, 0.32 * n + 2.0)  # clamp once, reuse everywhere below --
    # using the unclamped value in the fraction math (as an earlier version
    # did) desyncs the axes position from the actual figure size for small n
    fig = plt.figure(figsize=(6.5, fig_height), dpi=150)
    left, bottom, width, height = 0.30, 2.2 / fig_height, 0.65, 1 - 2.2 / fig_height - 0.10
    ax = fig.add_axes([left, bottom, width, height])
    ylim = (-1, n)
    x_a, x_b = 0, 1
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(*ylim)

    axis_height_points = height * fig.get_figheight() * 72
    row_spacing_points = axis_height_points / (ylim[1] - ylim[0])
    max_radius_points = row_spacing_points * 0.40
    min_radius_points = row_spacing_points * 0.12

    def size_scale(scores):
        frac = (scores - score_min) / score_range
        radius = min_radius_points + np.clip(frac, 0, 1) * (max_radius_points - min_radius_points)
        return np.pi * radius**2

    sc = ax.scatter([x_a] * n, y, s=size_scale(scores_a), c=log_tpm(tpm_a), cmap=cmap, norm=norm,
                     edgecolors="#4C72B0", linewidths=0.8, zorder=3)
    ax.scatter([x_b] * n, y, s=size_scale(scores_b), c=log_tpm(tpm_b), cmap=cmap, norm=norm,
               edgecolors="#4C72B0", linewidths=0.8, zorder=3)

    ax.set_xticks([x_a, x_b])
    ax.set_xticklabels(list(cell_types), fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(tf_names, fontsize=8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)

    # A fixed-inches colorbar axes, not fig.colorbar's automatic ax=/pad=
    # placement -- that pad is a FRACTION of the scatter axes' own height,
    # which varies a lot with n (few TFs -> tiny axes -> pad collapses to
    # nothing and the colorbar overlaps the x-tick labels). Placing it at
    # an absolute inches-from-bottom position keeps the gap correct
    # regardless of how many TFs are plotted.
    cbar_ax = fig.add_axes([left, 1.35 / fig_height, 0.65 * 0.6, 0.16 / fig_height])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("RNA expression (log10 TPM+1)", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def write_gc_comparison_plot(seqs_a, label_a, seqs_b, label_b, out_path, title):
    """Overlaid GC-content histograms for two sequence sets -- the visual
    check for comparing two design runs against each other (e.g. an
    activity-optimized set vs a specificity-optimized set for the same
    TF) without needing to pre-commit to one objective before seeing
    either. GC content is the one metric that's directly comparable
    across runs regardless of what each was optimized for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gc_a = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs_a]) if seqs_a else np.array([])
    gc_b = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs_b]) if seqs_b else np.array([])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bins = np.linspace(0.2, 0.8, 25)
    if len(gc_a):
        ax.hist(gc_a, bins=bins, color="#378ADD", alpha=0.55, label=f"{label_a} (n={len(gc_a)})", edgecolor="white")
    if len(gc_b):
        ax.hist(gc_b, bins=bins, color="#0F6E56", alpha=0.55, label=f"{label_b} (n={len(gc_b)})", edgecolor="white")
    ax.set_xlabel("GC content")
    ax.set_ylabel("Number of designed sequences")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
