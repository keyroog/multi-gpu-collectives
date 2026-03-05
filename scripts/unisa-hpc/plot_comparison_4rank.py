#!/usr/bin/env python3
"""
Genera grafici di confronto NCCL vs oneCCL vs RCCL per ogni coppia (collettiva, dtype).
Dati di riferimento: results/unisa-hpc/4_rank/
  - NCCL e oneCCL: 4 rank
  - RCCL:          8 rank su singolo nodo (nota esplicita nel grafico)

Note sul formato RCCL:
  - colonna tempo : time_ms_1coll  (vs time_ms di NCCL/oneCCL)
  - nomi collettive abbreviati: ar→allreduce, a2a→alltoall
  - dtype abbreviati: f→float, d→double, i→int
  - file flat in rccl/, non struttura per collettiva/size

Usage:
    python scripts/unisa-hpc/plot_comparison_4rank.py
    python scripts/unisa-hpc/plot_comparison_4rank.py --results-dir results/unisa-hpc/4_rank --out-dir plots/unisa-hpc/4_rank
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "lines.linewidth":    6,
    "lines.markersize":  28,
})

RCCL_COLLECTIVE_MAP = {
    "ar":     "allreduce",
    "a2a":    "alltoall",
    "bcast":  "broadcast",
    "reduce": "reduce",
    "gather": "gather",
    "ag":     "allgather",
}

RCCL_DTYPE_MAP = {
    "f": "float",
    "d": "double",
    "i": "int",
    "h": "half",
    "b": "bfloat16",
}


LIBRARY_STYLE = {
    "nccl":        {"color": "#2ca02c", "marker": "s", "label": "NCCL (4 ranks)"},
    "oneccl":      {"color": "#1f77b4", "marker": "o", "label": "oneCCL NVIDIA (4 ranks)"},
    "rccl":        {"color": "#d62728", "marker": "^", "label": "RCCL (8 ranks)"},
    "oneccl-amd":  {"color": "#ff7f0e", "marker": "D", "label": "oneCCL AMD (8 ranks)"},
}


def _fmt_size(b: int) -> str:
    for unit, thresh in [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)]:
        if b >= thresh and b % thresh == 0:
            return f"{b // thresh}{unit}"
    return f"{b}B"


def load_nccl_oneccl(results_dir: str) -> pd.DataFrame:
    """Carica tutti i CSV NCCL e oneCCL da results_dir/{nccl,oneccl}/{collective}/."""
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
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_flat_csvs(results_dir: str, subdir: str, library_name: str) -> pd.DataFrame:
    """Carica e normalizza CSV flat (formato RCCL/oneCCL-AMD) da results_dir/subdir/."""
    src_dir = os.path.join(results_dir, subdir)
    if not os.path.isdir(src_dir):
        return pd.DataFrame()

    frames = []
    for csv_path in glob.glob(os.path.join(src_dir, "*.csv")):
        try:
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                "global_rank":   "rank",
                "time_ms_1coll": "time_ms",
            })
            df["collective"] = df["collective"].map(lambda x: RCCL_COLLECTIVE_MAP.get(x, x))
            df["data_type"]  = df["data_type"].map(lambda x: RCCL_DTYPE_MAP.get(x, x))
            df["library"]    = library_name
            frames.append(df)
        except Exception as e:
            print(f"Warning: impossibile leggere {csv_path}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_results(results_dir: str) -> pd.DataFrame:
    main_df      = load_nccl_oneccl(results_dir)
    rccl_df      = load_flat_csvs(results_dir, "rccl",       "rccl")
    oneccl_amd   = load_flat_csvs(results_dir, "oneccl-amd", "oneccl-amd")
    frames = [df for df in (main_df, rccl_df, oneccl_amd) if not df.empty]
    if not frames:
        print(f"Nessun file CSV trovato in {results_dir}")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Mantieni solo le colonne necessarie per i plot
    cols = ["library", "collective", "data_type", "message_size_bytes", "num_ranks", "rank", "run_id", "time_ms"]
    return combined[[c for c in cols if c in combined.columns]]


def _mad(series: pd.Series) -> float:
    med = series.median()
    return float(np.median(np.abs(series - med)))


def _make_plot(grouped: pd.DataFrame, metric: str, ylabel: str,
               collective: str, dtype: str, suffix: str, out_dir: str,
               formats: list[str] | None = None, yscale: str = "log"):
    fig, ax = plt.subplots(figsize=(32, 20))

    for library, style in LIBRARY_STYLE.items():
        lib_data = grouped[grouped["library"] == library].sort_values("message_size_bytes")
        if lib_data.empty:
            continue
        ax.errorbar(
            lib_data["message_size_bytes"],
            lib_data["value"],
            yerr=lib_data["mad"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=6,
            markersize=24,
            capsize=10,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)

    present_sizes = sorted(grouped["message_size_bytes"].unique())
    ax.set_xticks(present_sizes)
    ax.set_xticklabels(
        [_fmt_size(int(s)) for s in present_sizes],
        rotation=45, ha="right", rotation_mode="anchor",
    )

    ax.set_xlabel("Message size")
    ax.set_ylabel(ylabel)
    ax.legend()
    grid_which = "both" if yscale == "log" else "major"
    ax.grid(True, which=grid_which, linestyle="--", alpha=0.4)

    fig.tight_layout()
    for fmt in (formats or ["png"]):
        out_path = os.path.join(out_dir, f"{collective}_{dtype}_{suffix}.{fmt}")
        kwargs = {"dpi": 150} if fmt != "pdf" else {}
        fig.savefig(out_path, **kwargs)
        print(f"Salvato: {out_path}")
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, collective: str, dtype: str, out_dir: str,
                    formats: list[str] | None = None, yscale: str = "log"):
    subset = df[(df["collective"] == collective) & (df["data_type"] == dtype)].copy()
    if subset.empty:
        return
    _sizes = sorted(subset["message_size_bytes"].unique())
    subset = subset[subset["message_size_bytes"] > _sizes[1]]
    if subset.empty:
        return

    # Per ogni run, prendi il max time_ms tra i rank (= wall-clock della collettiva)
    per_run = (
        subset
        .groupby(["library", "message_size_bytes", "run_id", "num_ranks"])["time_ms"]
        .max()
        .reset_index()
    )

    per_run["goodput_gbps"] = (
        (per_run["message_size_bytes"] / per_run["num_ranks"]) * 8 / (per_run["time_ms"] * 1e6)
    )

    # --- Grafico 1: tempo (ms) ---
    grp_time = (
        per_run
        .groupby(["library", "message_size_bytes"])["time_ms"]
        .agg(value="median", mad=_mad)
        .reset_index()
    )
    _make_plot(grp_time, "time_ms", "Time (ms)",
               collective, dtype, "time", out_dir, formats, yscale)

    # --- Grafico 2: goodput (Gb/s) ---
    grp_goodput = (
        per_run
        .groupby(["library", "message_size_bytes"])["goodput_gbps"]
        .agg(value="median", mad=_mad)
        .reset_index()
    )
    _make_plot(grp_goodput, "goodput_gbps", "Goodput (Gb/s)",
               collective, dtype, "goodput", out_dir, formats, yscale)


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici di confronto NCCL vs oneCCL vs RCCL (4-rank run)."
    )
    parser.add_argument(
        "--results-dir",
        default="results/unisa-hpc/4_rank",
        help="Directory radice dei risultati (default: results/unisa-hpc/4_rank)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/unisa-hpc/4_rank",
        help="Directory di output per i grafici (default: plots/unisa-hpc/4_rank)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Formato/i di output (default: png). Es: --format pdf  o  --format png pdf",
    )
    parser.add_argument(
        "--yscale",
        default="log",
        choices=["log", "linear"],
        help="Scala dell'asse Y (default: log)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all_results(args.results_dir)
    if df.empty:
        return

    collectives = sorted(df["collective"].unique())
    dtypes      = sorted(df["data_type"].unique())

    for collective in collectives:
        for dtype in dtypes:
            plot_comparison(df, collective, dtype, args.out_dir, args.formats, args.yscale)

    n_pairs = len(collectives) * len(dtypes)
    print(f"\nCompletato. {n_pairs} coppie, {n_pairs * 2} grafici in {args.out_dir}/")


if __name__ == "__main__":
    main()
