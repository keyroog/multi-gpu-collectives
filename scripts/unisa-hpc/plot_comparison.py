#!/usr/bin/env python3
"""
Genera grafici di confronto NCCL vs oneCCL per ogni coppia (collettiva, dtype).

Usage:
    python scripts/unisa-hpc/plot_comparison.py
    python scripts/unisa-hpc/plot_comparison.py --results-dir results/unisa-hpc --out-dir plots/unisa-hpc
"""
import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# Dimensioni target in bytes (quelle usate dallo script run_benchmark.sh)
TARGET_SIZES_BYTES = [
    1, 8, 64, 512,
    4 * 1024,          # 4 KiB
    32 * 1024,          # 32 KiB
    256 * 1024,         # 256 KiB
    2 * 1024**2,        # 2 MiB
    16 * 1024**2,       # 16 MiB
    128 * 1024**2,      # 128 MiB
    1 * 1024**3,        # 1 GiB
]

SIZE_LABELS = {
    1: "1B", 8: "8B", 64: "64B", 512: "512B",
    4096: "4KiB", 32768: "32KiB", 262144: "256KiB",
    2097152: "2MiB", 16777216: "16MiB", 134217728: "128MiB",
    1073741824: "1GiB",
}

LIBRARY_STYLE = {
    "nccl":   {"color": "#2ca02c", "marker": "s", "label": "NCCL"},
    "oneccl": {"color": "#1f77b4", "marker": "o", "label": "oneCCL"},
}


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Carica tutti i CSV da results_dir/{nccl,oneccl}/{collective}/."""
    frames = []
    for library in ("nccl", "oneccl"):
        lib_dir = os.path.join(results_dir, library)
        if not os.path.isdir(lib_dir):
            continue
        for collective in os.listdir(lib_dir):
            coll_dir = os.path.join(lib_dir, collective)
            if not os.path.isdir(coll_dir):
                continue
            for csv_path in glob.glob(os.path.join(coll_dir, "*_results.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    frames.append(df)
                except Exception as e:
                    print(f"Warning: impossibile leggere {csv_path}: {e}")
    if not frames:
        print(f"Nessun file CSV trovato in {results_dir}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _make_plot(grouped: pd.DataFrame, metric: str, ylabel: str,
               collective: str, dtype: str, suffix: str, out_dir: str):
    """Genera un singolo grafico per la metrica indicata."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for library, style in LIBRARY_STYLE.items():
        lib_data = grouped[grouped["library"] == library].sort_values("message_size_bytes")
        if lib_data.empty:
            continue
        ax.errorbar(
            lib_data["message_size_bytes"],
            lib_data["value"],
            yerr=lib_data["std"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=7,
            capsize=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("linear")

    present_sizes = sorted(grouped["message_size_bytes"].unique())
    ax.set_xticks(present_sizes)
    ax.set_xticklabels(
        [SIZE_LABELS.get(s, f"{s}B") for s in present_sizes],
        rotation=45, ha="right",
    )

    ax.set_xlabel("Message size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{collective} — {dtype}  (NCCL vs oneCCL)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{collective}_{dtype}_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


def plot_comparison(df: pd.DataFrame, collective: str, dtype: str, out_dir: str):
    """Genera due grafici (tempo e goodput) per la coppia (collective, dtype)."""
    subset = df[(df["collective"] == collective) & (df["data_type"] == dtype)].copy()
    if subset.empty:
        return

    # Filtra solo le size target
    subset = subset[subset["message_size_bytes"].isin(TARGET_SIZES_BYTES)]
    if subset.empty:
        return

    # Goodput in Gb/s per ogni misurazione
    subset["goodput_gbps"] = subset["message_size_bytes"] * 8 / (subset["time_ms"] * 1e6)

    # --- Grafico 1: tempo (ms) ---
    grp_time = (
        subset
        .groupby(["library", "message_size_bytes"])["time_ms"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_time.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_time, "time_ms", "Time (ms)",
               collective, dtype, "time", out_dir)

    # --- Grafico 2: goodput (Gb/s) ---
    grp_goodput = (
        subset
        .groupby(["library", "message_size_bytes"])["goodput_gbps"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_goodput.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_goodput, "goodput_gbps", "Goodput (Gb/s)",
               collective, dtype, "goodput", out_dir)

    # --- Grafico 3: initialization time (ms) ---
    grp_init = (
        subset
        .groupby(["library", "message_size_bytes"])["init_time_ms"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_init.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_init, "init_time_ms", "Initialization Time (ms)",
               collective, dtype, "init", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di confronto NCCL vs oneCCL per collettiva e dtype."
    )
    parser.add_argument(
        "--results-dir",
        default="results/unisa-hpc",
        help="Directory radice dei risultati (default: results/unisa-hpc)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/unisa-hpc",
        help="Directory di output per i grafici (default: plots/unisa-hpc)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all_results(args.results_dir)
    if df.empty:
        return

    # Genera un grafico per ogni coppia (collective, dtype)
    collectives = sorted(df["collective"].unique())
    dtypes = sorted(df["data_type"].unique())

    for collective in collectives:
        for dtype in dtypes:
            plot_comparison(df, collective, dtype, args.out_dir)

    n_pairs = len(collectives) * len(dtypes)
    print(f"\nCompletato. {n_pairs} coppie, {n_pairs * 3} grafici in {args.out_dir}/")


if __name__ == "__main__":
    main()
