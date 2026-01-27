"""
Plotting script for matmul accumulate core sweep results.
Compares Winograd and Base architectures with ratio plots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_file: str) -> dict:
    """Load results from JSON file."""
    results_path = Path(__file__).parent / results_file
    with open(results_path, "r") as f:
        return json.load(f)


def extract_metric_grid(
    data: dict,
    metric_name: str = "estimated_num_transistors",
    architecture: Optional[str] = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Extract a 2D grid of metric values for a specific architecture.

    Args:
        data: Loaded JSON data
        metric_name: Name of the metric to extract
        architecture: Architecture to filter by (None for legacy single-arch data)

    Returns:
        Tuple of (grid, dim_vals, width_vals)
    """
    dim_vals = sorted(data["sweep_config"]["dim_vals"])
    width_vals = sorted(data["sweep_config"]["width_vals"])

    grid = np.full((len(dim_vals), len(width_vals)), np.nan)

    for result in data["results"]:
        config = result["config"]
        metrics = result.get("metrics")

        # Filter by architecture if specified
        if architecture is not None:
            if config.get("architecture") != architecture:
                continue

        if metrics is None:
            continue

        dim_idx = dim_vals.index(config["dim_m"])
        width_idx = width_vals.index(config["a_width"])

        if metric_name in metrics:
            grid[dim_idx, width_idx] = metrics[metric_name]

    return grid, dim_vals, width_vals


def plot_metric_heatmap(
    grid: np.ndarray,
    dim_vals: list[int],
    width_vals: list[int],
    metric_name: str,
    output_file: Optional[str] = None,
    log_scale: bool = True,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot a heatmap of metric values.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle log scale for color normalization
    valid_vals = grid[~np.isnan(grid)]
    if len(valid_vals) == 0:
        print(f"Warning: No valid data for {metric_name}")
        return

    if log_scale and np.min(valid_vals) > 0:
        from matplotlib.colors import LogNorm
        norm = LogNorm(
            vmin=vmin if vmin else np.min(valid_vals),
            vmax=vmax if vmax else np.max(valid_vals)
        )
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(
            vmin=vmin if vmin else np.nanmin(grid),
            vmax=vmax if vmax else np.nanmax(grid)
        )

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(width_vals)))
    ax.set_xticklabels(width_vals)
    ax.set_yticks(range(len(dim_vals)))
    ax.set_yticklabels(dim_vals)

    ax.set_xlabel("Bit Width (a_width = b_width)")
    ax.set_ylabel("Dimension (dim_m = dim_n = dim_k)")

    if title is None:
        title = f"{metric_name.replace('_', ' ').title()}"
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name.replace("_", " ").title())

    # Add text annotations
    for i in range(len(dim_vals)):
        for j in range(len(width_vals)):
            value = grid[i, j]
            if not np.isnan(value):
                if value >= 1e6:
                    text = f"{value/1e6:.1f}M"
                elif value >= 1e3:
                    text = f"{value/1e3:.1f}K"
                elif value < 0.01:
                    text = f"{value:.3f}"
                elif value < 1:
                    text = f"{value:.2f}"
                else:
                    text = f"{value:.0f}"

                # Choose text color based on background
                text_color = "white" if value > np.nanmedian(valid_vals) else "black"
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="red", fontsize=9)

    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def plot_architecture_comparison(
    data: dict,
    metric_name: str = "estimated_num_transistors",
    output_file: Optional[str] = None,
) -> None:
    """
    Plot side-by-side comparison of architectures.
    """
    architectures = data["sweep_config"].get("architectures", ["winograd"])

    if len(architectures) < 2:
        print("Need at least 2 architectures for comparison")
        return

    fig, axes = plt.subplots(1, len(architectures) + 1, figsize=(6 * (len(architectures) + 1), 5))

    grids = {}
    dim_vals = None
    width_vals = None

    # Get data for each architecture
    for idx, arch in enumerate(architectures):
        grid, dim_vals, width_vals = extract_metric_grid(data, metric_name, architecture=arch)
        grids[arch] = grid

    # Find common scale
    all_valid = np.concatenate([g[~np.isnan(g)] for g in grids.values()])
    if len(all_valid) == 0:
        print(f"No valid data for {metric_name}")
        return

    vmin, vmax = np.min(all_valid), np.max(all_valid)

    from matplotlib.colors import LogNorm
    if vmin > 0:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Plot each architecture
    for idx, arch in enumerate(architectures):
        ax = axes[idx]
        grid = grids[arch]

        im = ax.imshow(grid, cmap="viridis", norm=norm, aspect="auto")

        ax.set_xticks(range(len(width_vals)))
        ax.set_xticklabels(width_vals)
        ax.set_yticks(range(len(dim_vals)))
        ax.set_yticklabels(dim_vals)

        ax.set_xlabel("Bit Width")
        ax.set_ylabel("Dimension")
        ax.set_title(f"{arch.title()}")

        # Add annotations
        for i in range(len(dim_vals)):
            for j in range(len(width_vals)):
                value = grid[i, j]
                if not np.isnan(value):
                    if value >= 1e6:
                        text = f"{value/1e6:.1f}M"
                    elif value >= 1e3:
                        text = f"{value/1e3:.1f}K"
                    else:
                        text = f"{value:.0f}"
                    ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", color="red", fontsize=8)

    # Plot ratio (winograd / base)
    if "winograd" in grids and "base" in grids:
        ax_ratio = axes[-1]
        ratio_grid = grids["winograd"] / grids["base"]

        # Use diverging colormap centered at 1
        from matplotlib.colors import TwoSlopeNorm, Normalize
        valid_ratios = ratio_grid[~np.isnan(ratio_grid)]
        if len(valid_ratios) > 0:
            vmin_r = np.min(valid_ratios)
            vmax_r = np.max(valid_ratios)
            # Center at 1.0, but handle cases where all values are on one side
            if vmin_r >= 1.0:
                # All ratios >= 1, use linear norm
                norm_ratio = Normalize(vmin=1.0, vmax=vmax_r)
            elif vmax_r <= 1.0:
                # All ratios <= 1, use linear norm
                norm_ratio = Normalize(vmin=vmin_r, vmax=1.0)
            else:
                # Values span both sides of 1
                norm_ratio = TwoSlopeNorm(vmin=vmin_r, vcenter=1.0, vmax=vmax_r)

            im_ratio = ax_ratio.imshow(ratio_grid, cmap="RdYlGn_r", norm=norm_ratio, aspect="auto")

            ax_ratio.set_xticks(range(len(width_vals)))
            ax_ratio.set_xticklabels(width_vals)
            ax_ratio.set_yticks(range(len(dim_vals)))
            ax_ratio.set_yticklabels(dim_vals)

            ax_ratio.set_xlabel("Bit Width")
            ax_ratio.set_ylabel("Dimension")
            ax_ratio.set_title("Ratio (Winograd / Base)")

            cbar = plt.colorbar(im_ratio, ax=ax_ratio)
            cbar.set_label("Ratio (<1 = Winograd better)")

            # Add annotations
            for i in range(len(dim_vals)):
                for j in range(len(width_vals)):
                    value = ratio_grid[i, j]
                    if not np.isnan(value):
                        text = f"{value:.2f}"
                        text_color = "black"
                        ax_ratio.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)
                    else:
                        ax_ratio.text(j, i, "N/A", ha="center", va="center", color="red", fontsize=9)
    else:
        axes[-1].set_visible(False)

    plt.suptitle(f"Architecture Comparison - {metric_name.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def plot_ratio_heatmap(
    data: dict,
    metric_name: str = "estimated_num_transistors",
    output_file: Optional[str] = None,
) -> None:
    """
    Plot just the ratio heatmap (Winograd / Base).
    """
    grid_winograd, dim_vals, width_vals = extract_metric_grid(data, metric_name, architecture="winograd")
    grid_base, _, _ = extract_metric_grid(data, metric_name, architecture="base")

    ratio_grid = grid_winograd / grid_base

    fig, ax = plt.subplots(figsize=(10, 8))

    valid_ratios = ratio_grid[~np.isnan(ratio_grid)]
    if len(valid_ratios) == 0:
        print(f"No valid ratio data for {metric_name}")
        return

    from matplotlib.colors import TwoSlopeNorm, Normalize
    vmin_r = np.min(valid_ratios)
    vmax_r = np.max(valid_ratios)
    # Handle cases where all values are on one side of 1.0
    if vmin_r >= 1.0:
        norm_ratio = Normalize(vmin=1.0, vmax=vmax_r)
    elif vmax_r <= 1.0:
        norm_ratio = Normalize(vmin=vmin_r, vmax=1.0)
    else:
        norm_ratio = TwoSlopeNorm(vmin=vmin_r, vcenter=1.0, vmax=vmax_r)

    im = ax.imshow(ratio_grid, cmap="RdYlGn_r", norm=norm_ratio, aspect="auto")

    ax.set_xticks(range(len(width_vals)))
    ax.set_xticklabels(width_vals)
    ax.set_yticks(range(len(dim_vals)))
    ax.set_yticklabels(dim_vals)

    ax.set_xlabel("Bit Width (a_width = b_width)")
    ax.set_ylabel("Dimension (dim_m = dim_n = dim_k)")
    ax.set_title(f"Winograd / Base Ratio - {metric_name.replace('_', ' ').title()}\n(<1 = Winograd better, >1 = Base better)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Ratio")

    # Add annotations
    for i in range(len(dim_vals)):
        for j in range(len(width_vals)):
            value = ratio_grid[i, j]
            if not np.isnan(value):
                text = f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="red", fontsize=10)

    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def plot_scaling_analysis(
    data: dict,
    output_file: Optional[str] = None,
) -> None:
    """
    Plot scaling analysis: transistor count vs dimensions for different widths.
    Compares both architectures.
    """
    architectures = data["sweep_config"].get("architectures", ["winograd"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"base": "tab:blue", "winograd": "tab:orange"}
    markers = {"base": "o", "winograd": "s"}

    # Plot 1: Transistor count vs dimension for each architecture/width
    ax1 = axes[0]
    for arch in architectures:
        grid, dim_vals, width_vals = extract_metric_grid(data, "estimated_num_transistors", architecture=arch)
        for j, width in enumerate(width_vals):
            values = grid[:, j]
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax1.plot(
                    np.array(dim_vals)[valid_mask],
                    values[valid_mask],
                    marker=markers.get(arch, "o"),
                    label=f"{arch} w={width}",
                    linestyle="-" if arch == "base" else "--",
                )

    ax1.set_xlabel("Dimension (dim_m = dim_n = dim_k)")
    ax1.set_ylabel("Estimated Transistor Count")
    ax1.set_title("Transistor Scaling with Dimension")
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ratio vs dimension for each width
    ax2 = axes[1]
    grid_winograd, dim_vals, width_vals = extract_metric_grid(data, "estimated_num_transistors", architecture="winograd")
    grid_base, _, _ = extract_metric_grid(data, "estimated_num_transistors", architecture="base")

    if grid_winograd is not None and grid_base is not None:
        ratio_grid = grid_winograd / grid_base
        for j, width in enumerate(width_vals):
            values = ratio_grid[:, j]
            valid_mask = ~np.isnan(values)
            if np.any(valid_mask):
                ax2.plot(
                    np.array(dim_vals)[valid_mask],
                    values[valid_mask],
                    marker="^",
                    label=f"width={width}",
                )

        ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Equal")
        ax2.set_xlabel("Dimension (dim_m = dim_n = dim_k)")
        ax2.set_ylabel("Winograd / Base Ratio")
        ax2.set_title("Architecture Ratio vs Dimension")
        ax2.set_xscale("log", base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Matmul Accumulate Core - Scaling Analysis", fontsize=14)
    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("--input", type=str, default="mmac_sweep_results.json", help="Input JSON file")
    args = parser.parse_args()

    results_file = args.input

    try:
        data = load_results(results_file)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        print("Please run eval_winograd_sweep.py first.")
        return

    print(f"Loaded results from: {results_file}")
    architectures = data["sweep_config"].get("architectures", ["winograd"])
    print(f"Architectures: {architectures}")
    print(f"Dimensions: {data['sweep_config']['dim_vals']}")
    print(f"Widths: {data['sweep_config']['width_vals']}")

    # Determine available metrics from first successful result
    available_metrics = []
    for result in data["results"]:
        if result.get("metrics"):
            available_metrics = list(result["metrics"].keys())
            break

    print(f"Available metrics: {available_metrics}")

    # Plot architecture comparison with ratio
    if len(architectures) >= 2:
        plot_architecture_comparison(
            data,
            "estimated_num_transistors",
            output_file="mmac_architecture_comparison.png",
        )

        # Plot standalone ratio heatmap
        plot_ratio_heatmap(
            data,
            "estimated_num_transistors",
            output_file="mmac_ratio_heatmap.png",
        )

    # Plot scaling analysis
    plot_scaling_analysis(data, output_file="mmac_scaling_analysis.png")


if __name__ == "__main__":
    main()
