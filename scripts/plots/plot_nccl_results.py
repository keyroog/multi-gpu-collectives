#!/usr/bin/env python3
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def human_readable_size(size):
    """Convert size in bytes to human-readable format."""
    if size < 1024:
        return f"{int(size)} B"
    for unit in ['KiB', 'MiB', 'GiB']:
        if size < 1024 ** 2:
            return f"{int(size / 1024)} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"

# Exact message sizes tested for plotting
PLOT_SIZES = [
    64,
    4 * 1024,
    256 * 1024,
    16 * 1024**2,
    256 * 1024**2,
    1 * 1024**3,  # 1 GiB
]

def load_data(logs_dir):
    dfs = []
    # Scan each collective folder under logs_dir
    for collective in os.listdir(logs_dir):
        collective_dir = os.path.join(logs_dir, collective)
        if not os.path.isdir(collective_dir):
            continue
        pattern = os.path.join(collective_dir, '*_results.csv')
        for filepath in glob.glob(pattern):
            try:
                df = pd.read_csv(filepath)
                df['collective'] = collective
                # Keep only relevant columns
                dfs.append(df[['collective','data_type','message_size_bytes','time_ms']])
            except Exception as e:
                print(f"Warning: failed to read {filepath}: {e}")
    if not dfs:
        print(f"No result files found in {logs_dir}")
        return None
    return pd.concat(dfs, ignore_index=True)

def plot_for_collective(df, collective, out_dir):
    dfc = df[df['collective'] == collective]
    # Compute mean time over runs and ranks
    grouped = dfc.groupby(['data_type','message_size_bytes'])['time_ms'].mean().reset_index()
    # Keep only the exact sizes tested
    grouped = grouped[grouped['message_size_bytes'].isin(PLOT_SIZES)]
    plt.figure()
    ax = sns.lineplot(data=grouped,
                      x='message_size_bytes',
                      y='time_ms',
                      hue='data_type',
                      marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.ylabel('Time (ms)')
    plt.title(f'NCCL {collective} performance')
    plt.legend(title='Data type')

    # Set human-readable labels for message sizes
    # Use predefined sizes for ticks
    ax.set_xticks(PLOT_SIZES)
    ax.set_xticklabels([human_readable_size(s) for s in PLOT_SIZES], rotation=45)
    plt.xlabel('Message size')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{collective}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate NCCL collective performance plots by message size and data type"
    )
    parser.add_argument(
        '--logs-dir',
        default='logs/nccl',
        help='Path to NCCL logs directory'
    )
    parser.add_argument(
        '--out-dir',
        default='scripts/plots',
        help='Output directory for plots'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(args.logs_dir)
    if df is None:
        return

    for collective in df['collective'].unique():
        plot_for_collective(df, collective, args.out_dir)

if __name__ == '__main__':
    main()