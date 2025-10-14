#!/usr/bin/env python3
"""
Matplotlib plotter for multiplier exploration results (single Parquet file).

Features:
- Scatter: any metric vs any metric (one point per configuration; includes switches_at_target)
- Line: metric vs sigma (lines grouped by run_id)
- Filters (categoricals + bit-width ranges)
- color_by + legend on/off
- 4 example plots (easy to extend by adding to PLOTS list)

Usage:
  python mpl_multiplier_plots.py --file data/multiplier_runs.parquet --out plots_mpl --legend on
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------- config ----------------------------- #

STATIC_METRICS = [
    "estimated_num_transistors",
    "transistor_count",
    "num_cells",
    "num_wires",
    "max_depth",
    "depth_y",
    "total_expr_nodes",
]

HOVER_BASE = [
    "design_label",
    "multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt",
    "a_enc", "b_enc", "y_enc",
    "n_bits", "a_w", "b_w", "y_w",
]


def design_label_from_row(row: pd.Series) -> str:
    def g(k: str, default: str = "NA"):
        return row[k] if (k in row.index and pd.notna(row[k])) else default
    return (
        f"{g('multiplier_opt')} | "
        f"PPG={g('ppg_opt')} | PPA={g('ppa_opt')} | FSA={g('fsa_opt')} | "
        f"enc={g('a_enc')}/{g('b_enc')}→{g('y_enc')}"
    )


def target_sigma_for(n_bits: int) -> float:
    return (3.0 / 16.0) * (2 ** int(n_bits))


def choose_area_column(df: pd.DataFrame) -> str:
    if "estimated_num_transistors" in df.columns:
        return "estimated_num_transistors"
    if "transistor_count" in df.columns:
        return "transistor_count"
    if "num_cells" in df.columns:
        return "num_cells"
    raise ValueError("No area-like column found (need one of: estimated_num_transistors, transistor_count, num_cells).")


# --------------------------- data helpers -------------------------- #

def load_df(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine="pyarrow").copy()

    # Derive n_bits if missing
    if "n_bits" not in df.columns:
        if "a_w" in df.columns:
            df["n_bits"] = df["a_w"].astype("int64")
        elif "y_w" in df.columns:
            df["n_bits"] = (df["y_w"].astype("int64") // 2).clip(lower=1)
        else:
            df["n_bits"] = 0

    for col in ["a_w", "b_w", "y_w"]:
        if col not in df.columns:
            df[col] = pd.Series([np.nan] * len(df), dtype="float64")

    # Label
    df["design_label"] = df.apply(design_label_from_row, axis=1).astype("string")

    # Types
    for c in ["multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt", "a_enc", "b_enc", "y_enc"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def compute_switches_at_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per configuration with `switches_at_target`."""
    key_cols = [
        "n_bits", "a_w", "b_w", "y_w",
        "multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt",
        "a_enc", "b_enc", "y_enc", "design_label",
    ]
    for k in key_cols:
        if k not in df.columns:
            df[k] = np.nan

    parts = []
    for _, g in df.groupby(key_cols, dropna=False):
        nbits_val = int(g["n_bits"].iloc[0]) if pd.notna(g["n_bits"].iloc[0]) else 0
        tgt = target_sigma_for(nbits_val)
        gg = g.copy()
        gg["sigma_dist"] = (gg["sigma"].astype(float) - tgt).abs()
        idx = gg["sigma_dist"].idxmin()
        best = gg.loc[[idx], key_cols + ["switches"]].rename(columns={"switches": "switches_at_target"})
        parts.append(best)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key_cols + ["switches_at_target"])


def build_design_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration (static metrics + switches_at_target)."""
    key_cols = [
        "n_bits", "a_w", "b_w", "y_w",
        "multiplier_opt", "ppg_opt", "ppa_opt", "fsa_opt",
        "a_enc", "b_enc", "y_enc", "design_label",
    ]
    static_cols = [c for c in STATIC_METRICS if c in df.columns]
    base = df[key_cols + static_cols].drop_duplicates(subset=key_cols).reset_index(drop=True)
    sat = compute_switches_at_target(df)
    out = pd.merge(base, sat, on=key_cols, how="left")
    return out


# ------------------------------ filters ---------------------------- #

@dataclass
class Filters:
    multiplier_opt: Optional[Sequence[str]] = None
    ppg_opt: Optional[Sequence[str]] = None
    ppa_opt: Optional[Sequence[str]] = None
    fsa_opt: Optional[Sequence[str]] = None
    a_enc: Optional[Sequence[str]] = None
    b_enc: Optional[Sequence[str]] = None
    y_enc: Optional[Sequence[str]] = None
    design_label: Optional[Sequence[str]] = None
    n_bits: Optional[Tuple[int, int]] = None
    a_w: Optional[Tuple[int, int]] = None
    b_w: Optional[Tuple[int, int]] = None
    y_w: Optional[Tuple[int, int]] = None


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()

    def filt(col, values):
        nonlocal out
        if values is not None and len(values) > 0 and col in out.columns:
            out = out[out[col].astype("string").isin(values)]

    filt("multiplier_opt", f.multiplier_opt)
    filt("ppg_opt", f.ppg_opt)
    filt("ppa_opt", f.ppa_opt)
    filt("fsa_opt", f.fsa_opt)
    filt("a_enc", f.a_enc)
    filt("b_enc", f.b_enc)
    filt("y_enc", f.y_enc)
    filt("design_label", f.design_label)

    def rng(col, r):
        nonlocal out
        if r is not None and col in out.columns:
            out = out[(out[col] >= r[0]) & (out[col] <= r[1])]

    rng("n_bits", f.n_bits)
    rng("a_w", f.a_w)
    rng("b_w", f.b_w)
    rng("y_w", f.y_w)

    return out


# ------------------------------ plotting --------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _slug(*parts: str) -> str:
    s = "_".join(p for p in parts if p).replace("/", "-").replace("→", "to").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))


def _category_colors(values: Sequence) -> Dict:
    """Map unique category values to matplotlib default cycle colors."""
    uniq = list(dict.fromkeys([str(v) for v in values]))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    colors = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
    return colors


def scatter_plot(
    design_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: Optional[str],
    legend: bool,
    title: str,
    outfile: str,
) -> None:
    if design_df.empty or x_metric not in design_df.columns or y_metric not in design_df.columns:
        print(f"[warn] scatter skipped: missing metric(s) or empty data ({x_metric}, {y_metric})")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_by and color_by in design_df.columns:
        vals = design_df[color_by].astype("string")
        cmap = _category_colors(vals)
        for val, g in design_df.groupby(vals):
            ax.scatter(g[x_metric], g[y_metric], label=str(val), s=28, alpha=0.85, color=cmap[str(val)])
    else:
        ax.scatter(design_df[x_metric], design_df[y_metric], s=28, alpha=0.85)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if legend and color_by and color_by in design_df.columns:
        ax.legend(fontsize=6, loc="best", frameon=False, markerscale=0.9, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def line_plot_sigma(
    full_df: pd.DataFrame,
    y_metric: str,
    color_by: Optional[str],
    legend: bool,
    title: str,
    outfile: str,
) -> None:
    """Lines connect ONLY within the same run_id."""
    if full_df.empty or "sigma" not in full_df.columns or y_metric not in full_df.columns:
        print(f"[warn] line skipped: missing 'sigma' or '{y_metric}' or empty data")
        return

    _ensure_dir(os.path.dirname(outfile) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Guarantee run_id exists
    if "run_id" not in full_df.columns:
        full_df = full_df.copy()
        full_df["run_id"] = (
            full_df["multiplier_opt"].astype(str) + "|" +
            full_df["ppg_opt"].astype(str) + "|" +
            full_df["ppa_opt"].astype(str) + "|" +
            full_df["fsa_opt"].astype(str) + "|" +
            full_df["a_enc"].astype(str) + "/" +
            full_df["b_enc"].astype(str) + "→" +
            full_df["y_enc"].astype(str)
        )

    # Prepare colors by category (one color per value of color_by)
    if color_by and color_by in full_df.columns:
        cat_vals_per_run = full_df.groupby("run_id")[color_by].first().astype("string")
        cmap = _category_colors(cat_vals_per_run.values)
        color_for_run = {rid: cmap[str(cat_vals_per_run.loc[rid])] for rid in cat_vals_per_run.index}
    else:
        color_for_run = {}

    # Plot each run as one line (sorted by sigma; unique per (run_id, sigma))
    for rid, g in full_df.groupby("run_id"):
        gg = (g.dropna(subset=["sigma", y_metric]) # error
                .sort_values("sigma")
                .drop_duplicates(subset=["run_id", "sigma"], keep="first"))
        if gg.empty:
            continue
        color = color_for_run.get(rid, None)
        label = None
        if legend and color_by and color_by in full_df.columns:
            label = str(cat_vals_per_run.loc[rid])
        ax.plot(gg["sigma"], gg[y_metric], marker="o", markersize=3.5, linewidth=1.2, alpha=0.9,
                color=color, label=label)

    ax.set_xlabel("sigma")
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if legend and color_by and color_by in full_df.columns:
        # Deduplicate identical legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if uniq:
            ax.legend(*zip(*uniq), fontsize=6, loc="best", frameon=False, markerscale=0.9, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ------------------------- example orchestration -------------------- #

@dataclass
class PlotConfig:
    kind: str                  # 'scatter' or 'line'
    title: str
    filename: str
    x_metric: Optional[str] = None    # scatter only
    y_metric: Optional[str] = None    # scatter + line (line uses only y_metric)
    color_by: Optional[str] = None
    legend: bool = True
    #filters: Filters = Filters()
    #use default factory to avoid mutable default
    filters: Filters = field(default_factory=Filters)


def run_plot(cfg: PlotConfig, full_df: pd.DataFrame, design_df: pd.DataFrame, out_dir: str) -> None:
    out_file = os.path.join(out_dir, _slug(cfg.filename))
    if cfg.kind == "scatter":
        ddf = apply_filters(design_df, cfg.filters)
        if cfg.x_metric is None or cfg.y_metric is None:
            print(f"[warn] scatter '{cfg.title}' skipped: x_metric/y_metric not set")
            return
        scatter_plot(ddf, cfg.x_metric, cfg.y_metric, cfg.color_by, cfg.legend, cfg.title, out_file)
    elif cfg.kind == "line":
        fdf = apply_filters(full_df, cfg.filters)
        if cfg.y_metric is None:
            print(f"[warn] line '{cfg.title}' skipped: y_metric not set")
            return
        line_plot_sigma(fdf, cfg.y_metric, cfg.color_by, cfg.legend, cfg.title, out_file)
    else:
        print(f"[warn] unknown plot kind: {cfg.kind}")


# ------------------------------- main ------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to single Parquet file (pandas-written).")
    ap.add_argument("--out", default="plots_mpl", help="Output directory for PNG files.")
    ap.add_argument("--legend", choices=["on", "off"], default="on", help="Show legends globally.")
    args = ap.parse_args()

    df = load_df(args.file)
    design_df = build_design_level_table(df)

    # Resolve a sensible area metric for examples
    try:
        area_col = choose_area_column(design_df)
    except Exception:
        area_col = None

    # -------------------- 4 example plots -------------------- #
    PLOTS: List[PlotConfig] = []

    # 1) Scatter: switches_at_target vs area, colored by n_bits
    if area_col:
        PLOTS.append(PlotConfig(
            kind="scatter",
            title=f"switches_at_target vs {area_col} (one point per configuration)",
            filename=f"scatter_switches_at_target_vs_{area_col}.png",
            x_metric=area_col,
            y_metric="switches_at_target",
            color_by="n_bits",
            legend=(args.legend == "on"),
            filters=Filters(),  # add your filters here
        ))

    # 2) Scatter: switches_at_target vs max_depth (fallback to total_expr_nodes), colored by multiplier_opt
    depth_y = "max_depth" if "max_depth" in design_df.columns else ("total_expr_nodes" if "total_expr_nodes" in design_df.columns else None)
    if depth_y:
        PLOTS.append(PlotConfig(
            kind="scatter",
            title=f"switches_at_target vs {depth_y}",
            filename=f"scatter_switches_at_target_vs_{depth_y}.png",
            x_metric=depth_y,
            y_metric="switches_at_target",
            color_by="multiplier_opt",
            legend=(args.legend == "on"),
            filters=Filters(),  # e.g., Filters(a_enc=["unsigned"], b_enc=["unsigned"])
        ))

    # 3) Line: switches vs sigma, colored by n_bits (lines connect per run_id)
    if "switches" in df.columns:
        PLOTS.append(PlotConfig(
            kind="line",
            title="switches vs sigma (lines per run_id)",
            filename="line_switches_vs_sigma.png",
            y_metric="switches",
            color_by="n_bits",
            legend=(args.legend == "on"),
            filters=Filters(),  # add e.g., Filters(n_bits=(8, 16))
        ))

    # 4) Line: switches vs sigma, colored by multiplier_opt (example range filter reserved)
    if "switches" in df.columns:
        # You can narrow the range by editing Filters(n_bits=(min,max), ...)
        PLOTS.append(PlotConfig(
            kind="line",
            title="switches vs sigma (colored by multiplier_opt)",
            filename="line_switches_vs_sigma_by_multiplier_opt.png",
            y_metric="switches",
            color_by="multiplier_opt",
            legend=(args.legend == "on"),
            filters=Filters(),  # e.g., Filters(ppg_opt=["BASIC","BOOTH_OPTIMISED_SIGNED"])
        ))

    # Run all examples
    for cfg in PLOTS:
        run_plot(cfg, df, design_df, args.out)

    print(f"[done] Wrote plots to: {os.path.abspath(args.out)}")
    print("Tip: add more PlotConfig entries to the PLOTS list or generate them programmatically.")


if __name__ == "__main__":
    main()
