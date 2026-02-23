#!/usr/bin/env python3
"""
Grafico a barre dell'initialization time: NCCL vs oneCCL × 4 rank vs 8 rank.

Legge i file *_init_time_*ranks_results.csv generati da build/{lib}/init_time.
Usa max_init_ms (tempo collettivo = rank più lento), iter >= 1 (warmup escluso).
Aggrega per mediana; error bar = IQR (25°–75° percentile).

Usage:
    python scripts/unisa-hpc/plot_init_time.py
    python scripts/unisa-hpc/plot_init_time.py \
        --results-base results/unisa-hpc --out-dir plots/unisa-hpc
"""

import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    bar_width = 0.55
    gap = 0.25          # spazio extra tra coppie nccl / oneccl
    positions = []
    x = 0.0
    for i, key in enumerate(ORDER):
        if i == 2:       # separazione visiva tra NCCL e oneCCL
            x += gap
        positions.append(x)
        x += bar_width + 0.15

    found_any = False
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
            linewidth=0.7,
            yerr=[[median - q25], [q75 - median]],
            error_kw=dict(ecolor="black", elinewidth=1.2, capsize=5),
            zorder=2,
        )
        ax.text(
            pos, q75 + max(df["max_init_ms"]) * 0.02,
            f"{median:.1f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    if not found_any:
        print("Nessun dato da plottare.")
        plt.close(fig)
        return

    # asse x: etichette
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [BAR_STYLE[k][2] for k in ORDER],
        fontsize=8.5,
    )

    ax.set_ylabel("Communicator Init Time (ms)", fontsize=10)
    ax.set_title("Initialization Time — NCCL vs oneCCL", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(-bar_width * 0.8, positions[-1] + bar_width * 0.8)

    # nota error bar
    ax.text(
        0.98, 0.97, "Error bar: IQR (25°–75°)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="#555555",
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "init_time_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")

# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", default="results/unisa-hpc")
    parser.add_argument("--out-dir",      default="plots/unisa-hpc")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.results_base)
    if df.empty:
        print("Nessun file *_init_time_*ranks_results.csv trovato.")
        return

    # riepilogo dati caricati
    print(df.groupby(["library", "num_ranks"])["max_init_ms"]
          .agg(n="count", median="median", std="std")
          .round(3)
          .to_string())
    print()

    plot(df, args.out_dir)

if __name__ == "__main__":
    main()
