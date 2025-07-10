#!/usr/bin/env python3
"""
Script per generare grafici da benchmark CSV in logs/nccl
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni directory
BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / 'logs' / 'nccl'
PLOT_DIR = BASE / 'plots'

sns.set(style='whitegrid')

# Operazioni da processare
operations = ['allgather', 'allreduce', 'alltoall', 'broadcast', 'gather', 'reduce', 'reduce_scatter', 'scatter']

def load_and_process(file_path):
    df = pd.read_csv(file_path)
    # Se non esiste bandwidth calcolala (bytes/time_us -> MB/s)
    if 'bandwidth' not in df.columns and 'time' in df.columns and 'size' in df.columns:
        df['bandwidth'] = df['size'] / df['time'] * 1e6 / (1024**2)
    return df

for op in operations:
    op_log = LOG_DIR / op
    if not op_log.exists():
        continue
    # directory di output
    out_dir = PLOT_DIR / op
    out_dir.mkdir(parents=True, exist_ok=True)

    # carica tutti i risultati
    data = {}
    for csv in op_log.glob('*_results.csv'):
        parts = csv.stem.split('_')
        dtype = parts[2]
        df = load_and_process(csv)
        data.setdefault(dtype, []).append(df)

    # concatene e ordina per ciascun dtype
    for dtype, dfs in data.items():
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values('size')
        # plot bandwidth
        plt.figure()
        sns.lineplot(x='size', y='bandwidth', data=df_all, marker='o')
        plt.xscale('log')
        plt.xlabel('Message Size [B]')
        plt.ylabel('Bandwidth [MB/s]')
        plt.title(f'{op} - {dtype} - Bandwidth')
        plt.savefig(out_dir / f'{op}_{dtype}_bandwidth.png')
        plt.close()

        # plot time
        plt.figure()
        sns.lineplot(x='size', y='time', data=df_all, marker='o')
        plt.xscale('log')
        plt.xlabel('Message Size [B]')
        plt.ylabel('Time [us]')
        plt.title(f'{op} - {dtype} - Time per call')
        plt.savefig(out_dir / f'{op}_{dtype}_time.png')
        plt.close()

    # confronto float vs double
    if 'float' in data and 'double' in data:
        df_f = pd.concat(data['float'], ignore_index=True).sort_values('size')
        df_d = pd.concat(data['double'], ignore_index=True).sort_values('size')
        plt.figure()
        sns.lineplot(x='size', y='bandwidth', data=df_f, marker='o', label='float')
        sns.lineplot(x='size', y='bandwidth', data=df_d, marker='o', label='double')
        plt.xscale('log')
        plt.xlabel('Message Size [B]')
        plt.ylabel('Bandwidth [MB/s]')
        plt.title(f'{op} - Confronto float vs double')
        plt.legend()
        plt.savefig(out_dir / f'{op}_float_vs_double_bandwidth.png')
        plt.close()

# Plot di sintesi in summary
title = 'Riepilogo performance operazioni'
plt.figure(figsize=(10, 6))
for op in operations:
    for dtype in ['float', 'double']:
        csvs = list((LOG_DIR / op).glob(f'*{dtype}*_results.csv'))
        if not csvs:
            continue
        dfs = [load_and_process(c) for c in csvs]
        df_all = pd.concat(dfs, ignore_index=True).sort_values('size')
        sns.lineplot(x='size', y='bandwidth', data=df_all, label=f'{op}-{dtype}')

plt.xscale('log')
plt.xlabel('Message Size [B]')
plt.ylabel('Bandwidth [MB/s]')
plt.title(title)
plt.legend(loc='best')
out = PLOT_DIR / 'summary'
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / 'summary_all_bandwidth.png')
plt.close()

print('Plot generati in', PLOT_DIR)
