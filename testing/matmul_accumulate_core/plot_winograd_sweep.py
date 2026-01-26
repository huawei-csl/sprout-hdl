"""
Plotting script for Winograd matmul accumulate core sweep results.
Generates visual grid plots for transistor count and other metrics.
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
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Extract a 2D grid of metric values.

    Args:
        data: Loaded JSON data
        metric_name: Name of the metric to extract

    Returns:
        Tuple of (grid, dim_vals, width_vals)
    """
    dim_vals = sorted(data["sweep_config"]["dim_vals"])
    width_vals = sorted(data["sweep_config"]["width_vals"])

    grid = np.full((len(dim_vals), len(width_vals)), np.nan)

    for result in data["results"]:
        config = result["config"]
        metrics = result.get("metrics")

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
) -> None:
    """
    Plot a heatmap of metric values.

    Args:
        grid: 2D array of metric values (dim x width)
        dim_vals: List of dimension values
        width_vals: List of width values
        metric_name: Name of the metric (for labeling)
        output_file: Path to save the plot
        log_scale: Whether to use log scale for colors
        title: Optional title override
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle log scale for color normalization
    if log_scale and np.nanmin(grid) > 0:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    else:
        norm = None

    im = ax.imshow(grid, cmap="viridis", norm=norm, aspect="auto")

    # Set tick labels
    ax.set_xticks(range(len(width_vals)))
    ax.set_xticklabels(width_vals)
    ax.set_yticks(range(len(dim_vals)))
    ax.set_yticklabels(dim_vals)

    ax.set_xlabel("Bit Width (a_width = b_width)")
    ax.set_ylabel("Dimension (dim_m = dim_n = dim_k)")

    if title is None:
        title = f"Winograd Matmul Core - {metric_name.replace('_', ' ').title()}"
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name.replace("_", " ").title())

    # Add text annotations
    for i in range(len(dim_vals)):
        for j in range(len(width_vals)):
            value = grid[i, j]
            if not np.isnan(value):
                # Format large numbers
                if value >= 1e6:
                    text = f"{value/1e6:.1f}M"
                elif value >= 1e3:
                    text = f"{value/1e3:.1f}K"
                else:
                    text = f"{value:.0f}"

                # Choose text color based on background
                bg_val = value if not log_scale else np.log10(value)
                threshold = (np.nanmax(grid if not log_scale else np.log10(grid[grid > 0])) +
                            np.nanmin(grid if not log_scale else np.log10(grid[grid > 0]))) / 2
                text_color = "white" if bg_val > threshold else "black"

                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="red", fontsize=9)

    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def plot_multiple_metrics(
    data: dict,
    metrics: list[str],
    output_file: Optional[str] = None,
) -> None:
    """
    Plot multiple metrics as subplots.

    Args:
        data: Loaded JSON data
        metrics: List of metric names to plot
        output_file: Path to save the plot
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        grid, dim_vals, width_vals = extract_metric_grid(data, metric_name)

        # Handle log scale
        from matplotlib.colors import LogNorm
        valid_vals = grid[~np.isnan(grid) & (grid > 0)]
        if len(valid_vals) > 0:
            norm = LogNorm(vmin=np.min(valid_vals), vmax=np.max(valid_vals))
        else:
            norm = None

        im = ax.imshow(grid, cmap="viridis", norm=norm, aspect="auto")

        ax.set_xticks(range(len(width_vals)))
        ax.set_xticklabels(width_vals)
        ax.set_yticks(range(len(dim_vals)))
        ax.set_yticklabels(dim_vals)

        ax.set_xlabel("Bit Width")
        ax.set_ylabel("Dimension")
        ax.set_title(metric_name.replace("_", " ").title())

        plt.colorbar(im, ax=ax)

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

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Winograd Matmul Accumulate Core - Parameter Sweep", fontsize=14)
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
    """
    grid, dim_vals, width_vals = extract_metric_grid(data, "estimated_num_transistors")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Transistor count vs dimension for each width
    for j, width in enumerate(width_vals):
        values = grid[:, j]
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax1.plot(
                np.array(dim_vals)[valid_mask],
                values[valid_mask],
                marker="o",
                label=f"width={width}",
            )

    ax1.set_xlabel("Dimension (dim_m = dim_n = dim_k)")
    ax1.set_ylabel("Estimated Transistor Count")
    ax1.set_title("Transistor Scaling with Dimension")
    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Transistor count vs width for each dimension
    for i, dim in enumerate(dim_vals):
        values = grid[i, :]
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            ax2.plot(
                np.array(width_vals)[valid_mask],
                values[valid_mask],
                marker="s",
                label=f"dim={dim}",
            )

    ax2.set_xlabel("Bit Width (a_width = b_width)")
    ax2.set_ylabel("Estimated Transistor Count")
    ax2.set_title("Transistor Scaling with Bit Width")
    ax2.set_yscale("log")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Winograd Matmul Core - Scaling Analysis", fontsize=14)
    plt.tight_layout()

    if output_file:
        output_path = Path(__file__).parent / output_file
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    results_file = "winograd_sweep_results.json"

    try:
        data = load_results(results_file)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        print("Please run eval_winograd_sweep.py first.")
        return

    print(f"Loaded results from: {results_file}")
    print(f"Sweep config: dim_vals={data['sweep_config']['dim_vals']}, "
          f"width_vals={data['sweep_config']['width_vals']}")

    # Determine available metrics from first successful result
    available_metrics = []
    for result in data["results"]:
        if result.get("metrics"):
            available_metrics = list(result["metrics"].keys())
            break

    print(f"Available metrics: {available_metrics}")

    # Plot transistor count heatmap
    grid, dim_vals, width_vals = extract_metric_grid(data, "estimated_num_transistors")
    plot_metric_heatmap(
        grid, dim_vals, width_vals,
        "estimated_num_transistors",
        output_file="winograd_transistor_heatmap.png",
    )

    # Plot scaling analysis
    plot_scaling_analysis(data, output_file="winograd_scaling_analysis.png")

    # Plot multiple metrics if available
    key_metrics = ["estimated_num_transistors", "num_cells", "num_wires"]
    metrics_to_plot = [m for m in key_metrics if m in available_metrics]
    if len(metrics_to_plot) > 1:
        plot_multiple_metrics(data, metrics_to_plot, output_file="winograd_multi_metrics.png")


if __name__ == "__main__":
    main()
