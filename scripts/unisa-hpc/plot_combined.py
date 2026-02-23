#!/usr/bin/env python3
"""
Genera grafici combinati 4-in-1 con tutte le curve:
{NCCL, oneCCL} × {4 rank, 8 rank} sullo stesso grafico.

Per ogni coppia (collettiva, dtype) produce un grafico con 4 curve
che permette il confronto completo tra librerie e configurazioni.

Usage:
    python scripts/unisa-hpc/plot_combined.py
    python scripts/unisa-hpc/plot_combined.py --results-base results/unisa-hpc --out-dir plots/unisa-hpc/combined
"""
import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt

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

# 4 combinazioni: libreria × num_ranks
COMBINED_STYLE = {
    ("nccl", 4):   {"color": "#2ca02c", "marker": "s", "linestyle": "-",  "label": "NCCL — 4 rank (1 nodo)"},
    ("nccl", 8):   {"color": "#98df8a", "marker": "s", "linestyle": "--", "label": "NCCL — 8 rank (2 nodi)"},
    ("oneccl", 4): {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "oneCCL — 4 rank (1 nodo)"},
    ("oneccl", 8): {"color": "#aec7e8", "marker": "o", "linestyle": "--", "label": "oneCCL — 8 rank (2 nodi)"},
}


def load_results(results_dir: str, num_ranks: int) -> pd.DataFrame:
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
    combined = pd.concat(frames, ignore_index=True)
    combined["num_ranks_config"] = num_ranks
    return combined


def _make_plot(grouped: pd.DataFrame, ylabel: str,
               collective: str, dtype: str, suffix: str, out_dir: str):
    """Genera un singolo grafico combinato 4-in-1."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for (library, n_ranks), style in COMBINED_STYLE.items():
        data = grouped[
            (grouped["library"] == library)
            & (grouped["num_ranks_config"] == n_ranks)
        ].sort_values("message_size_bytes")
        if data.empty:
            continue
        ax.errorbar(
            data["message_size_bytes"],
            data["value"],
            yerr=data["std"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
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
    ax.set_title(f"{collective} — {dtype}  (NCCL vs oneCCL × 4 rank vs 8 rank)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{collective}_{dtype}_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


def plot_combined(df: pd.DataFrame, collective: str, dtype: str, out_dir: str):
    """Genera grafici combinati 4-in-1 per la coppia (collective, dtype)."""
    subset = df[
        (df["collective"] == collective) & (df["data_type"] == dtype)
    ].copy()
    if subset.empty:
        return

    subset = subset[subset["message_size_bytes"].isin(TARGET_SIZES_BYTES)]
    if subset.empty:
        return

    subset["goodput_gbps"] = subset["message_size_bytes"] * 8 / (subset["time_ms"] * 1e6)

    # --- Grafico 1: tempo (ms) ---
    grp_time = (
        subset
        .groupby(["library", "num_ranks_config", "message_size_bytes"])["time_ms"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_time.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_time, "Time (ms)",
               collective, dtype, "time", out_dir)

    # --- Grafico 2: goodput (Gb/s) ---
    grp_goodput = (
        subset
        .groupby(["library", "num_ranks_config", "message_size_bytes"])["goodput_gbps"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_goodput.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_goodput, "Goodput (Gb/s)",
               collective, dtype, "goodput", out_dir)

    # --- Grafico 3: initialization time (ms) ---
    grp_init = (
        subset
        .groupby(["library", "num_ranks_config", "message_size_bytes"])["init_time_ms"]
        .agg(["median", "std"])
        .reset_index()
    )
    grp_init.rename(columns={"median": "value"}, inplace=True)
    _make_plot(grp_init, "Initialization Time (ms)",
               collective, dtype, "init", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici combinati {NCCL, oneCCL} × {4 rank, 8 rank}."
    )
    parser.add_argument(
        "--results-base",
        default="results/unisa-hpc",
        help="Directory base contenente 4_rank/ e 8_rank/ (default: results/unisa-hpc)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/unisa-hpc/combined",
        help="Directory di output per i grafici (default: plots/unisa-hpc/combined)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Carica e unisci i dati da entrambe le configurazioni
    df_4 = load_results(os.path.join(args.results_base, "4_rank"), num_ranks=4)
    df_8 = load_results(os.path.join(args.results_base, "8_rank"), num_ranks=8)

    frames = [d for d in (df_4, df_8) if not d.empty]
    if not frames:
        print("Nessun dato trovato.")
        return
    df = pd.concat(frames, ignore_index=True)

    collectives = sorted(df["collective"].unique())
    dtypes = sorted(df["data_type"].unique())

    count = 0
    for collective in collectives:
        for dtype in dtypes:
            plot_combined(df, collective, dtype, args.out_dir)
            count += 1

    print(f"\nCompletato. {count} coppie, {count * 3} grafici in {args.out_dir}/")


if __name__ == "__main__":
    main()
