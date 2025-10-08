#!/usr/bin/env python3
"""
Plot multiplier exploration results from a single Parquet file (pandas-only).

Generates PNGs (no on-screen windows):
- sigma_vs_switches/sigma_vs_switches_nbits<...>.png            (legend inside, tiny font)
- area_vs_depth/area_vs_<depth>[_nbits<...>].png                (legend inside, tiny font)
- switches_at_target_vs_metric/switches_at_target_vs_<metric>[_nbits<...>].png
  (NO Pareto here anymore)
- switches_at_target_pareto/pareto_area_vs_switches_targetsigma_nbits<...>.png
- switches_at_target_pareto/pareto_depth_vs_switches_targetsigma_<depth>_nbits<...>.png

Usage:
  python plot_multiplier_results.py \
    --file data/multiplier_runs.parquet \
    --out plots \
    --metrics estimated_num_transistors,max_depth,num_cells \
    --legend on
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


DEFAULT_METRICS: List[str] = [
    "estimated_num_transistors",
    "transistor_count",
    "num_cells",
    "num_wires",
    "max_depth",
    "depth_y",
    "total_expr_nodes",
]

BASE_COLS = [
    "run_id",
    "timestamp",
    "module_name",
    "n_bits",
    "sigma",
    "switches",
    "multiplier_cls",
    "ppg_opt",
    "ppa_opt",
    "fsa_opt",
    "a_enc",
    "b_enc",
    "y_enc",
    "a_w",
    "b_w",
    "y_w",
    "num_vectors",
    "total_expr_nodes",
    "max_depth",
    "depth_y",
    "num_wires",
    "num_cells",
    "estimated_num_transistors",
    "transistor_count",
    "op_nodes_json",
]


# ----------------------------- utilities ----------------------------- #


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def design_label_from_row(row: pd.Series) -> str:
    def g(k: str, default: str = "NA"):
        return row[k] if k in row.index and pd.notna(row[k]) else default

    return f"{g('multiplier_cls')} | " f"PPG={g('ppg_opt')} | PPA={g('ppa_opt')} | FSA={g('fsa_opt')} | " f"enc={g('a_enc')}/{g('b_enc')}→{g('y_enc')}"


def add_design_label(df: pd.DataFrame) -> pd.DataFrame:
    if "design_label" not in df.columns:
        df = df.copy()
        df["design_label"] = df.apply(design_label_from_row, axis=1)
    return df


def load_parquet_single(file_path: str, metrics: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine="pyarrow")
    want = [c for c in dict.fromkeys(BASE_COLS + metrics) if c in df.columns]
    return df[want].copy()


def group_sigma_mean(df: pd.DataFrame, ycols: List[str]) -> pd.DataFrame:
    grp_cols = ["n_bits", "design_label", "sigma"]
    existing_y = [c for c in ycols if c in df.columns]
    if not existing_y:
        return df.drop_duplicates(subset=grp_cols)
    agg = df.groupby(grp_cols, as_index=False)[existing_y].mean(numeric_only=True)
    return agg


def choose_area_column(df: pd.DataFrame) -> str:
    if "estimated_num_transistors" in df.columns:
        return "estimated_num_transistors"
    if "transistor_count" in df.columns:
        return "transistor_count"
    if "num_cells" in df.columns:
        return "num_cells"
    raise ValueError("No area-like column found (need one of: estimated_num_transistors, transistor_count, num_cells).")


def nearest_sigma_rows(df: pd.DataFrame, target_sigma: float) -> pd.DataFrame:
    df = df.copy()
    df["sigma_dist"] = (df["sigma"].astype(float) - float(target_sigma)).abs()
    idx = df.groupby(["n_bits", "design_label"])["sigma_dist"].idxmin()
    out = df.loc[idx].drop(columns=["sigma_dist"])
    return out


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str, minimize: Tuple[bool, bool] = (True, True)) -> pd.DataFrame:
    asc_x, asc_y = minimize
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return d
    d = d.sort_values([x_col, y_col], ascending=[asc_x, asc_y]).reset_index(drop=True)
    pf_idx = []
    best_y = None
    for i, y in enumerate(d[y_col].to_numpy()):
        if best_y is None:
            pf_idx.append(i)
            best_y = y
        else:
            if asc_y and y < best_y:
                pf_idx.append(i)
                best_y = y
            elif (not asc_y) and y > best_y:
                pf_idx.append(i)
                best_y = y
    return d.iloc[pf_idx, :]


def maybe_legend(ax, show: bool, fontsize: int = 3, **kwargs):
    if show:
        ax.legend(fontsize=fontsize, loc="best", frameon=False, markerscale=0.8, handlelength=0.8, **kwargs)


# ----------------------------- plotting ----------------------------- #


def lineplots_sigma_vs_switches(df: pd.DataFrame, outdir: str, show_legend: bool) -> None:
    """sigma vs switches (one figure per bitwidth), legend inside with tiny font."""
    ensure_dir(outdir)
    for nbits, dsub in df.groupby("n_bits"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for label, g in dsub.groupby("design_label"):
            g2 = g.sort_values("sigma")
            ax.plot(g2["sigma"], g2["switches"], label=label, linewidth=1.2)
        ax.set_xlabel("sigma")
        ax.set_ylabel("switches")
        ax.set_title(f"sigma vs switches (n_bits={nbits})")
        ax.grid(True, linestyle="--", alpha=0.4)
        maybe_legend(ax, show_legend, fontsize=3)
        fout = os.path.join(outdir, f"sigma_vs_switches_nbits{nbits}.png")
        fig.tight_layout()
        fig.savefig(fout, dpi=150)
        plt.close(fig)


def scatter_area_vs_depth(df: pd.DataFrame, outdir: str, show_legend: bool) -> None:
    """
    Scatter plots of area vs depth (both total and y-depth if available).
    One per bitwidth + one all-in. Legend inside with tiny font.
    """
    ensure_dir(outdir)
    area_col = choose_area_column(df)
    depth_cols = [c for c in ["max_depth", "depth_y"] if c in df.columns]

    # Reduce to design-unique static points (avoid sigma duplicates)
    static_cols = ["n_bits", "design_label", area_col] + depth_cols
    dstatic = df.drop_duplicates(subset=static_cols)[static_cols]

    def scatter_one(dsub: pd.DataFrame, suffix: str) -> None:
        for depth_col in depth_cols:
            if depth_col not in dsub.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, g in dsub.groupby("design_label"):
                ax.scatter(g[area_col], g[depth_col], s=20, alpha=0.9, label=label)
            ax.set_xlabel(area_col)
            ax.set_ylabel(depth_col)
            ax.set_title(f"Area vs {depth_col}{suffix}")
            ax.grid(True, linestyle="--", alpha=0.4)
            maybe_legend(ax, show_legend, fontsize=3)
            fout = os.path.join(outdir, f"area_vs_{depth_col}{suffix}.png")
            fig.tight_layout()
            fig.savefig(fout, dpi=150)
            plt.close(fig)

    # Per-bitwidth
    for nbits, g in dstatic.groupby("n_bits"):
        scatter_one(g, suffix=f"_nbits{nbits}")

    # All together
    scatter_one(dstatic, suffix="")


def scatter_switches_at_target_vs_metric(df: pd.DataFrame, metric: str, outdir: str, show_legend: bool) -> None:
    """
    For each bitwidth compute target sigma = 3/16 * 2**n_bits,
    pick nearest-sigma row per design, and scatter: switches vs metric.
    NO Pareto here anymore.
    """
    ensure_dir(outdir)

    # Combined plot (legend by n_bits, inside)
    fout_all = os.path.join(outdir, f"switches_at_target_vs_{metric}_ALL.png")
    fig_all, ax_all = plt.subplots(figsize=(8, 6))

    for nbits, dsub in df.groupby("n_bits"):
        sigma_target = (3.0 / 16.0) * (2 ** int(nbits))
        chosen = nearest_sigma_rows(dsub, sigma_target)
        if chosen.empty or metric not in chosen.columns:
            continue

        # per-bitwidth figure with legend per design INSIDE
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, g in chosen.groupby("design_label"):
            ax.scatter(g["switches"], g[metric], s=25, alpha=0.9, label=label)
        ax.set_xlabel("switches @ target sigma")
        ax.set_ylabel(metric)
        ax.set_title(f"Switches vs {metric} @ σ≈{sigma_target:.3f} (n_bits={nbits})")
        ax.grid(True, linestyle="--", alpha=0.4)
        maybe_legend(ax, show_legend, fontsize=3)
        fout = os.path.join(outdir, f"switches_at_target_vs_{metric}_nbits{nbits}.png")
        fig.tight_layout()
        fig.savefig(fout, dpi=150)
        plt.close(fig)

        # Add to combined figure (legend by bitwidth)
        ax_all.scatter(chosen["switches"], chosen[metric], s=20, alpha=0.6, label=f"n_bits={nbits}")

    ax_all.set_xlabel("switches @ target sigma")
    ax_all.set_ylabel(metric)
    ax_all.set_title(f"Switches vs {metric} @ target sigma (all bitwidths)")
    ax_all.grid(True, linestyle="--", alpha=0.4)
    maybe_legend(ax_all, True, fontsize=8)  # keep bitwidth legend visible
    fig_all.tight_layout()
    fig_all.savefig(fout_all, dpi=150)
    plt.close(fig_all)


def pareto_switches_at_target(df: pd.DataFrame, outdir: str, show_legend: bool) -> None:
    """
    Combined Pareto generation (Area vs Switches) and (Depth vs Switches) @ target sigma.
    All outputs go into `outdir` (e.g., switches_at_target_pareto/).
    """
    ensure_dir(outdir)
    depth_cols = [c for c in ["max_depth", "depth_y"] if c in df.columns]

    for nbits, dsub in df.groupby("n_bits"):
        sigma_target = (3.0 / 16.0) * (2 ** int(nbits))
        chosen = nearest_sigma_rows(dsub, sigma_target)
        if chosen.empty:
            continue

        # --- Area vs Switches ---
        area_col = None
        try:
            area_col = choose_area_column(chosen)
        except ValueError:
            pass

        if area_col is not None:
            d2 = chosen[["design_label", "switches", area_col]].dropna()
            if not d2.empty:
                pf = pareto_front(d2, x_col=area_col, y_col="switches", minimize=(True, True))
                fig, ax = plt.subplots(figsize=(8, 6))
                for label, g in d2.groupby("design_label"):
                    ax.scatter(g[area_col], g["switches"], s=20, alpha=0.7, label=label)
                if not pf.empty:
                    ax.plot(pf[area_col], pf["switches"], linewidth=2.0, label="Pareto front")
                ax.set_xlabel(area_col)
                ax.set_ylabel("switches @ target sigma")
                ax.set_title(f"Pareto (Area vs Switches) @ σ≈{sigma_target:.3f} (n_bits={nbits})")
                ax.grid(True, linestyle="--", alpha=0.4)
                maybe_legend(ax, show_legend, fontsize=3)
                fout = os.path.join(outdir, f"pareto_area_vs_switches_targetsigma_nbits{nbits}.png")
                fig.tight_layout()
                fig.savefig(fout, dpi=150)
                plt.close(fig)

        # --- Depth vs Switches ---
        for depth_col in depth_cols:
            d3 = chosen[["design_label", "switches", depth_col]].dropna()
            if d3.empty:
                continue
            pf = pareto_front(d3, x_col=depth_col, y_col="switches", minimize=(True, True))
            fig, ax = plt.subplots(figsize=(8, 6))
            for label, g in d3.groupby("design_label"):
                ax.scatter(g[depth_col], g["switches"], s=20, alpha=0.7, label=label)
            if not pf.empty:
                ax.plot(pf[depth_col], pf["switches"], linewidth=2.0, label="Pareto front")
            ax.set_xlabel(depth_col)
            ax.set_ylabel("switches @ target sigma")
            ax.set_title(f"Pareto (Depth vs Switches) @ σ≈{sigma_target:.3f} (n_bits={nbits})")
            ax.grid(True, linestyle="--", alpha=0.4)
            maybe_legend(ax, show_legend, fontsize=3)
            fout = os.path.join(outdir, f"pareto_depth_vs_switches_targetsigma_{depth_col}_nbits{nbits}.png")
            fig.tight_layout()
            fig.savefig(fout, dpi=150)
            plt.close(fig)


# ----------------------------- main ----------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to Parquet file (single file, pandas-written).")
    ap.add_argument("--out", default="plots", help="Output directory for PNG files.")
    ap.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics for switches-at-target-vs-metric.",
    )
    ap.add_argument("--no-pareto", action="store_true", help="Disable Pareto plots.")
    ap.add_argument("--legend", choices=["on", "off"], default="on", help="Show legends globally (default: on).")
    args = ap.parse_args()

    out_root = args.out
    ensure_dir(out_root)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    show_legend = args.legend == "on"

    print(f"[info] Loading Parquet: {args.file}")
    df = load_parquet_single(args.file, metrics)

    if "n_bits" not in df.columns:
        if "a_w" in df.columns:
            df["n_bits"] = df["a_w"].astype("int64")
        elif "y_w" in df.columns:
            df["n_bits"] = (df["y_w"].astype("int64") // 2).clip(lower=1)
        else:
            df["n_bits"] = 0

    df = add_design_label(df)

    # Aggregate to mean per (n_bits, design_label, sigma) for clean lines/scatters
    ycols = list(set(["switches"] + metrics))
    dline = group_sigma_mean(df, ycols=ycols)

    # A) sigma vs switches
    out_sig_sw = os.path.join(out_root, "sigma_vs_switches")
    lineplots_sigma_vs_switches(dline, outdir=out_sig_sw, show_legend=show_legend)

    # B) area vs depth (legend inside)
    out_scatter_ad = os.path.join(out_root, "area_vs_depth")
    try:
        scatter_area_vs_depth(df, outdir=out_scatter_ad, show_legend=show_legend)
    except ValueError as e:
        print(f"[warn] Area vs depth plots skipped: {e}")

    # C) switches@target vs metric (NO Pareto here)
    out_scatter_sm = os.path.join(out_root, "switches_at_target_vs_metric")
    for metric in metrics:
        if metric not in df.columns:
            print(f"[warn] Skipping switches@target scatter; column missing: {metric}")
            continue
        scatter_switches_at_target_vs_metric(dline, metric, outdir=out_scatter_sm, show_legend=show_legend)

    # D) Combined Pareto generation (Area & Depth vs Switches) @ target sigma
    if not args.no_pareto:
        out_pareto = os.path.join(out_root, "switches_at_target_pareto")
        pareto_switches_at_target(df, outdir=out_pareto, show_legend=show_legend)

    print(f"[done] Plots written under: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()
