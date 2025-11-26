#!/usr/bin/env python3
"""
Plots for floating-point multiplier evaluation results produced by
low_level_arithmetic/fp_multiplier_eval/run_fp_multiplier_eval.py.

Generates:
- Line: switches vs sigma (one line per run_id/config)
- Bar: AIG size (num_aig_gates) per config (dedup by run_id)
- Bar: AIG depth (aig_depth) per config (dedup by run_id)

Usage:
  python plot_fp_multiplier_eval.py --file data/fp_multiplier_runs_YYYYMMDD_HHMMSS.parquet --out plots_mpl
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# --------------------------- helpers --------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _category_colors(values) -> Dict[str, str]:
    uniq = list(dict.fromkeys([str(v) for v in values]))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = [f"C{i}" for i in range(10)]
    return {u: palette[i % len(palette)] for i, u in enumerate(uniq)}


def _fmt_label(row: pd.Series) -> str:
    kind = str(row.get("fmt_kind", "?"))
    if kind.lower() == "ieee":
        ew = row.get("EW", None)
        fw = row.get("FW", None)
        subn = row.get("subnormals", None)
        bits = 1 + (int(ew) if pd.notna(ew) else 0) + (int(fw) if pd.notna(fw) else 0)
        sn = "SN" if bool(subn) else "noSN"
        return f"IEEE E{int(ew)}M{int(fw)} FP{bits} {sn}"
    if kind.lower() == "hif8":
        return "HiF8"
    return kind


def load_df(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path, engine="pyarrow").copy()

    # Fill missing columns harmlessly
    for c in ["run_id", "sigma", "switches", "num_aig_gates", "aig_depth", "fmt_kind", "EW", "FW", "subnormals", "module_name", "dist"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Build a config label per row; dedup later by run_id
    df["config_label"] = df.apply(_fmt_label, axis=1).astype("string")
    return df


def build_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run_id with static metrics and a human label."""
    if df.empty:
        return df
    keep = [
        "run_id",
        "fmt_kind", "EW", "FW", "subnormals", "module_name", "dist",
        "num_aig_gates", "aig_depth",
        "config_label",
    ]
    present = [c for c in keep if c in df.columns]
    base = df[present].drop_duplicates(subset=["run_id"]).reset_index(drop=True)
    # Sort labels for nicer bar order
    return base.sort_values(["fmt_kind", "config_label"], na_position="last").reset_index(drop=True)


# --------------------------- plotting --------------------------- #

def plot_switches_vs_sigma(full_df: pd.DataFrame, out_file: str, color_by: str = "config_label") -> None:
    if full_df.empty:
        print("[warn] switches vs sigma skipped: empty data")
        return

    _ensure_dir(os.path.dirname(out_file) or ".")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine colors per run (based on chosen color_by column's first value per run)
    if color_by not in full_df.columns:
        full_df = full_df.copy()
        full_df[color_by] = full_df.get("fmt_kind", "?").astype("string")

    first_val = full_df.groupby("run_id")[color_by].first().astype("string")
    cmap = _category_colors(first_val.values)
    color_for_run = {rid: cmap[str(first_val.loc[rid])] for rid in first_val.index}

    for rid, g in full_df.groupby("run_id"):
        gg = (g.dropna(subset=["sigma", "switches"])  # ensure we have both
                .sort_values("sigma")
                .drop_duplicates(subset=["run_id", "sigma"], keep="first"))
        if gg.empty:
            continue
        color = color_for_run.get(rid)
        label = str(first_val.loc[rid])
        ax.semilogx(gg["sigma"], gg["switches"], marker="o", markersize=3.5, linewidth=1.2, alpha=0.9,
                color=color, label=label)

    ax.set_xlabel("Sigma of input vectors")
    ax.set_ylabel("Average Switches")
    ax.set_title("switches vs sigma (lines per run_id)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        ax.legend(*zip(*uniq), fontsize=6, loc="best", frameon=False, markerscale=0.9, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_bar_metric_per_config(config_df: pd.DataFrame, metric: str, out_file: str, title: str) -> None:
    if config_df.empty or metric not in config_df.columns:
        print(f"[warn] bar plot skipped: empty data or missing '{metric}'")
        return

    _ensure_dir(os.path.dirname(out_file) or ".")
    fig, ax = plt.subplots(figsize=(6, 3))

    # Sort by metric descending for readability
    df = config_df.dropna(subset=[metric]) #.sort_values(metric, ascending=False)
    labels = df["config_label"].astype("string").tolist()
    vals = df[metric].astype(float).tolist()

    ax.bar(range(len(vals)), vals, color="C0", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ------------------------------ main ------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to Parquet file produced by run_fp_multiplier_eval.py")
    ap.add_argument("--out", default="plots_mpl", help="Output directory for PNG files.")
    args = ap.parse_args()

    df = load_df(args.file)
    cfg_df = build_config_table(df)

    # 1) switches vs sigma lines (one line per run_id)
    plot_switches_vs_sigma(df, os.path.join(args.out, "line_switches_vs_sigma_fp.png"), color_by="config_label")

    # 2) Bar plots: AIG size and depth per configuration (dedup by run_id)
    plot_bar_metric_per_config(cfg_df, "num_aig_gates", os.path.join(args.out, "bar_num_aig_gates_fp.png"),
                               title="AIG Size (num_aig_gates) per configuration")
    plot_bar_metric_per_config(cfg_df, "aig_depth", os.path.join(args.out, "bar_aig_depth_fp.png"),
                               title="AIG Depth per configuration")

    print(f"[done] Wrote plots to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

