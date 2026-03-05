#!/usr/bin/env python3
"""
Grafico a barre dell'initialization time: NCCL vs oneCCL × 4 rank vs 8 rank.

Legge i file *_init_time_*ranks_results.csv generati da build/{lib}/init_time.
Usa max_init_ms (tempo collettivo = rank più lento), iter >= 1 (warmup escluso).
Aggrega per mediana; error bar = IQR (25°–75° percentile).

Usage:
    python scripts/python/plot_init_time.py
    python scripts/python/plot_init_time.py \
        --results-base results/unisa-hpc --out-dir plots/unisa-hpc
"""

import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size":         70,
    "axes.labelsize":    76,
    "xtick.labelsize":   66,
    "ytick.labelsize":   66,
    "legend.fontsize":   66,
    "xtick.major.pad":   16,
    "ytick.major.pad":   16,
    "xtick.major.size":  16,
    "ytick.major.size":  16,
    "xtick.major.width":  3,
    "ytick.major.width":  3,
    "axes.linewidth":     3,
    "axes.labelpad":     20,
    "legend.framealpha": 0.9,
})

# ---- Stile barre -------------------------------------------------------------
#   (library, num_ranks) -> (color, hatch, label)
BAR_STYLE = {
    ("nccl",   4): ("#2ca02c", "",   "NCCL — 4 rank\n(intra-nodo)"),
    ("oneccl", 4): ("#1f77b4", "",   "oneCCL — 4 rank\n(intra-nodo)"),
    ("nccl",   8): ("#2ca02c", "//", "NCCL — 8 rank\n(inter-nodo)"),
    ("oneccl", 8): ("#1f77b4", "//", "oneCCL — 8 rank\n(inter-nodo)"),
}
ORDER = [("nccl", 4), ("oneccl", 4), ("nccl", 8), ("oneccl", 8)]

# ---- Caricamento -------------------------------------------------------------

def load(results_base: str) -> pd.DataFrame:
    frames = []
    for rank_dir in glob.glob(os.path.join(results_base, "*_rank")):
        num_ranks = int(os.path.basename(rank_dir).replace("_rank", ""))
        for lib_dir in glob.glob(os.path.join(rank_dir, "*")):
            if not os.path.isdir(lib_dir):
                continue
            for csv_path in glob.glob(
                os.path.join(lib_dir, "*_init_time_*ranks_results.csv")
            ):
                try:
                    df = pd.read_csv(csv_path)
                    # iter >= 1: warmup non loggato, iter=0 è ccl::init di oneCCL
                    df = df[df["iter"] >= 1]
                    # una riga per iter: max_init_ms è identico per tutti i rank
                    df = df[df["rank"] == 0][["library", "iter", "max_init_ms"]].copy()
                    df["num_ranks"] = num_ranks
                    frames.append(df)
                except Exception as e:
                    print(f"  Warning: {csv_path}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ---- Plot --------------------------------------------------------------------

def plot(df: pd.DataFrame, out_dir: str, formats: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(32, 20))

    bar_width = 0.55
    gap = 0.25
    positions = []
    x = 0.0
    for i, key in enumerate(ORDER):
        if i == 2:
            x += gap
        positions.append(x)
        x += bar_width + 0.15

    found_any = False
    max_val = df["max_init_ms"].max()
    for pos, key in zip(positions, ORDER):
        lib, nranks = key
        sub = df[(df["library"] == lib) & (df["num_ranks"] == nranks)]["max_init_ms"]
        if sub.empty:
            continue
        found_any = True

        median = sub.median()
        q25, q75 = sub.quantile(0.25), sub.quantile(0.75)
        style = BAR_STYLE[key]

        ax.bar(
            pos, median, bar_width,
            color=style[0],
            hatch=style[1],
            edgecolor="black",
            linewidth=3,
            yerr=[[median - q25], [q75 - median]],
            error_kw=dict(ecolor="black", elinewidth=5, capsize=20),
            zorder=2,
        )
        ax.text(
            pos, q75 + max_val * 0.02,
            f"{median:.1f} ms",
            ha="center", va="bottom", fontsize=64, fontweight="bold",
        )

    if not found_any:
        print("Nessun dato da plottare.")
        plt.close(fig)
        return

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [BAR_STYLE[k][2] for k in ORDER],
        fontsize=48,
    )

    ax.set_ylabel("Communicator Init Time (ms)", fontsize=70)
    ax.set_title("Initialization Time — NCCL vs oneCCL", fontsize=72)
    ax.set_ylim(bottom=0, top=max_val * 1.35)
    ax.tick_params(axis="y", labelsize=60)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(-bar_width * 0.8, positions[-1] + bar_width * 0.8)

    ax.text(
        0.98, 0.97, "Error bar: IQR (25°–75°)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=52, color="#555555",
    )

    fig.tight_layout()
    for fmt in (formats or ["png"]):
        out_path = os.path.join(out_dir, f"init_time_comparison.{fmt}")
        kwargs = {"dpi": 150} if fmt != "pdf" else {}
        fig.savefig(out_path, **kwargs)
        print(f"Salvato: {out_path}")
    plt.close(fig)

# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera grafico a barre initialization time NCCL vs oneCCL."
    )
    parser.add_argument("--results-base", default="results/unisa-hpc",
                        help="Directory base contenente *_rank/ (default: results/unisa-hpc)")
    parser.add_argument("--out-dir", default="plots/unisa-hpc",
                        help="Directory di output (default: plots/unisa-hpc)")
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Formato/i di output (default: png)",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.results_base)
    if df.empty:
        print("Nessun file *_init_time_*ranks_results.csv trovato.")
        return

    print(df.groupby(["library", "num_ranks"])["max_init_ms"]
          .agg(n="count", median="median", std="std")
          .round(3)
          .to_string())
    print()

    plot(df, args.out_dir, args.formats)

if __name__ == "__main__":
    main()
