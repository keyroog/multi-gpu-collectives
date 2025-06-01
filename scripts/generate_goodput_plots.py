#!/usr/bin/env python3
"""
Script avanzato per analisi goodput dai risultati dei benchmark.
Analizza worst-rank timing, environment impact, e consistenza tra esecuzioni.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def load_results_with_environment(results_dir):
    """Carica tutti i file CSV con supporto per environment tracking."""
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {results_dir}")
        return None
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Retrocompatibilità: aggiungi colonne mancanti
            if 'run_number' not in df.columns:
                df['run_number'] = range(1, len(df) + 1)
                print(f"Aggiunta colonna run_number ai dati legacy: {file}")
            
            if 'environment' not in df.columns:
                df['environment'] = 'legacy'
                print(f"Aggiunta colonna environment ai dati legacy: {file}")
            
            dataframes.append(df)
            print(f"Caricato: {file} ({len(df)} righe)")
        except Exception as e:
            print(f"Errore nel caricare {file}: {e}")
    
    if not dataframes:
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def calculate_goodput_metrics(df):
    """Calcola metriche goodput per ogni configurazione."""
    # Raggruppa per run_number e configurazione, poi calcola worst-rank time
    goodput_data = []
    
    grouped = df.groupby(['library', 'collective', 'data_type', 'message_size_elements', 'run_number'])
    
    for name, group in grouped:
        library, collective, data_type, msg_size, run_num = name
        
        # Calcola goodput (worst-rank time) per questa esecuzione
        worst_time = group['time_ms'].max()
        avg_time = group['time_ms'].mean()
        min_time = group['time_ms'].min()
        num_ranks = group['num_ranks'].iloc[0]
        environment = group['environment'].iloc[0] if 'environment' in group.columns else 'unknown'
        
        goodput_data.append({
            'library': library,
            'collective': collective,
            'data_type': data_type,
            'message_size_elements': msg_size,
            'run_number': run_num,
            'goodput_time_ms': worst_time,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'time_spread_ms': worst_time - min_time,
            'num_ranks': num_ranks,
            'environment': environment
        })
    
    return pd.DataFrame(goodput_data)

def plot_goodput_analysis(goodput_df, output_dir):
    """Crea grafici di analisi goodput."""
    
    libraries = goodput_df['library'].unique()
    collectives = goodput_df['collective'].unique()
    data_types = goodput_df['data_type'].unique()
    
    for lib in libraries:
        for collective in collectives:
            for dtype in data_types:
                subset = goodput_df[
                    (goodput_df['library'] == lib) & 
                    (goodput_df['collective'] == collective) & 
                    (goodput_df['data_type'] == dtype)
                ]
                
                if subset.empty:
                    continue
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{lib.upper()} {collective.capitalize()} ({dtype}) - Goodput Analysis', fontsize=16)
                
                # 1. Goodput vs Message Size
                ax1 = axes[0, 0]
                msg_sizes = sorted(subset['message_size_elements'].unique())
                goodput_stats = subset.groupby('message_size_elements')['goodput_time_ms'].agg(['mean', 'std', 'min', 'max'])
                
                ax1.errorbar(goodput_stats.index, goodput_stats['mean'], yerr=goodput_stats['std'], 
                           marker='o', capsize=5, label='Goodput (worst-rank)', linewidth=2)
                
                # Confronta con average timing
                avg_stats = subset.groupby('message_size_elements')['avg_time_ms'].mean()
                ax1.plot(avg_stats.index, avg_stats.values, marker='s', label='Average time', linewidth=2, alpha=0.7)
                
                ax1.set_xlabel('Message Size (elements)')
                ax1.set_ylabel('Time (ms)')
                ax1.set_title('Goodput vs Average Timing')
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Run Consistency
                ax2 = axes[0, 1]
                for msg_size in msg_sizes[:3]:  # Solo prime 3 message sizes per chiarezza
                    size_data = subset[subset['message_size_elements'] == msg_size]
                    if len(size_data) > 1:
                        ax2.plot(size_data['run_number'], size_data['goodput_time_ms'], 
                               marker='o', label=f'{msg_size} elements', alpha=0.7)
                
                ax2.set_xlabel('Run Number')
                ax2.set_ylabel('Goodput Time (ms)')
                ax2.set_title('Goodput Consistency Across Runs')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # 3. Time Spread Analysis
                ax3 = axes[1, 0]
                spread_stats = subset.groupby('message_size_elements')['time_spread_ms'].agg(['mean', 'std'])
                ax3.bar(range(len(spread_stats)), spread_stats['mean'], yerr=spread_stats['std'], 
                       alpha=0.7, capsize=5)
                ax3.set_xlabel('Message Size')
                ax3.set_ylabel('Time Spread (max-min) ms')
                ax3.set_title('Rank-to-Rank Time Variation')
                ax3.set_xticks(range(len(spread_stats)))
                ax3.set_xticklabels([f'{int(x)}' for x in spread_stats.index], rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # 4. Environment Impact (se disponibile)
                ax4 = axes[1, 1]
                if subset['environment'].nunique() > 1:
                    env_data = subset.groupby(['environment', 'message_size_elements'])['goodput_time_ms'].mean().unstack()
                    env_data.plot(kind='bar', ax=ax4, alpha=0.7)
                    ax4.set_title('Environment Impact on Goodput')
                    ax4.set_ylabel('Goodput Time (ms)')
                    ax4.set_xlabel('Environment Configuration')
                else:
                    # Se non ci sono environment diversi, mostra distribuzione goodput
                    ax4.hist(subset['goodput_time_ms'], bins=20, alpha=0.7, edgecolor='black')
                    ax4.set_title('Goodput Time Distribution')
                    ax4.set_xlabel('Goodput Time (ms)')
                    ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Salva il grafico
                filename = f'{lib}_{collective}_{dtype}_goodput_analysis.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Grafico goodput salvato: {filepath}")
                plt.close()

def plot_goodput_comparison(goodput_df, output_dir):
    """Confronta goodput tra diverse configurazioni."""
    
    plt.figure(figsize=(14, 10))
    
    # Pivot per confronto tra data types
    pivot_data = goodput_df.pivot_table(
        index='message_size_elements',
        columns='data_type',
        values='goodput_time_ms',
        aggfunc='mean'
    )
    
    # Plot confronto data types
    for dtype in pivot_data.columns:
        plt.plot(pivot_data.index, pivot_data[dtype], 
                marker='o', label=f'{dtype} (goodput)', linewidth=3, markersize=8)
    
    plt.xlabel('Message Size (elements)')
    plt.ylabel('Goodput Time (ms) - Worst Rank')
    plt.title('Goodput Comparison: Worst-Rank Timing Analysis')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aggiungi annotations per identificare problemi di performance
    for dtype in pivot_data.columns:
        max_idx = pivot_data[dtype].idxmax()
        max_val = pivot_data[dtype].max()
        plt.annotate(f'Peak: {max_val:.2f}ms', 
                    xy=(max_idx, max_val),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    filename = 'goodput_comparison_all_types.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Grafico confronto goodput salvato: {filepath}")
    plt.close()

def generate_goodput_report(goodput_df, output_dir):
    """Genera report dettagliato di analisi goodput."""
    
    report_path = os.path.join(output_dir, 'goodput_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== GOODPUT ANALYSIS REPORT ===\n\n")
        f.write("This report analyzes worst-rank timing (goodput) across multiple benchmark runs.\n")
        f.write("Goodput represents the actual performance experienced by applications,\n")
        f.write("as collective operations complete when the slowest rank finishes.\n\n")
        
        f.write(f"Total benchmark runs analyzed: {goodput_df['run_number'].nunique()}\n")
        f.write(f"Configurations tested: {len(goodput_df)}\n")
        f.write(f"Data types: {', '.join(goodput_df['data_type'].unique())}\n")
        f.write(f"Message sizes: {', '.join(map(str, sorted(goodput_df['message_size_elements'].unique())))}\n\n")
        
        # Goodput summary per data type
        f.write("=== GOODPUT SUMMARY BY DATA TYPE ===\n")
        for dtype in goodput_df['data_type'].unique():
            dtype_data = goodput_df[goodput_df['data_type'] == dtype]
            f.write(f"\n{dtype.upper()}:\n")
            f.write(f"  Average goodput: {dtype_data['goodput_time_ms'].mean():.3f} ± {dtype_data['goodput_time_ms'].std():.3f} ms\n")
            f.write(f"  Best goodput: {dtype_data['goodput_time_ms'].min():.3f} ms\n")
            f.write(f"  Worst goodput: {dtype_data['goodput_time_ms'].max():.3f} ms\n")
            f.write(f"  Average time spread: {dtype_data['time_spread_ms'].mean():.3f} ms\n")
            
            # Trova configurazione con miglior goodput
            best_config = dtype_data.loc[dtype_data['goodput_time_ms'].idxmin()]
            f.write(f"  Best configuration: {best_config['message_size_elements']} elements, run #{best_config['run_number']}\n")
        
        # Analisi consistency
        f.write("\n=== CONSISTENCY ANALYSIS ===\n")
        consistency_stats = goodput_df.groupby(['data_type', 'message_size_elements'])['goodput_time_ms'].agg(['std', 'mean'])
        consistency_stats['cv'] = consistency_stats['std'] / consistency_stats['mean']  # Coefficient of variation
        
        f.write("Configurations with highest variation (CV > 0.1):\n")
        high_variation = consistency_stats[consistency_stats['cv'] > 0.1].sort_values('cv', ascending=False)
        if len(high_variation) > 0:
            for (dtype, msg_size), row in high_variation.head(5).iterrows():
                f.write(f"  {dtype}, {msg_size} elements: CV = {row['cv']:.3f}\n")
        else:
            f.write("  No configurations show high variation (all CV < 0.1)\n")
        
        # Environment impact
        f.write("\n=== ENVIRONMENT IMPACT ===\n")
        unique_envs = goodput_df['environment'].unique()
        if len(unique_envs) > 1:
            env_impact = goodput_df.groupby('environment')['goodput_time_ms'].agg(['mean', 'std', 'count'])
            f.write("Performance by environment configuration:\n")
            for env, stats in env_impact.iterrows():
                f.write(f"  {env}: {stats['mean']:.3f} ± {stats['std']:.3f} ms (n={stats['count']})\n")
        else:
            f.write("Single environment configuration used in all tests.\n")
        
        # Recommendations
        f.write("\n=== RECOMMENDATIONS ===\n")
        f.write("1. Focus on goodput (worst-rank) timing for real application performance\n")
        f.write("2. Investigate configurations with high time spread between ranks\n")
        f.write("3. Run multiple iterations to understand performance consistency\n")
        f.write("4. Consider GPU topology and Xe Link connectivity for Intel Max GPUs\n")
        f.write("5. Monitor environment variables impact on performance variations\n")
        
        # GPU Topology Notes
        f.write("\n=== GPU TOPOLOGY INVESTIGATION NOTES ===\n")
        f.write("For Intel GPU Max systems, investigate:\n")
        f.write("- Use 'ocloc query' to see available GPU devices and capabilities\n")
        f.write("- Check if GPUs are on same 'MACRO GPU' for optimal Xe Link connectivity\n")
        f.write("- Verify memory bandwidth and interconnect topology\n")
        f.write("- Consider NUMA affinity for multi-GPU setups\n")
        f.write("- Monitor ZE_DEBUG=1 output for Level-Zero device information\n")
    
    print(f"Report goodput salvato: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analisi goodput avanzata dai risultati benchmark')
    parser.add_argument('--input', required=True, help='Directory contenente i file CSV dei risultati')
    parser.add_argument('--output', required=True, help='Directory dove salvare i grafici e report')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], 
                       help='Formato dei grafici (default: png)')
    
    args = parser.parse_args()
    
    # Crea directory di output se non esiste
    os.makedirs(args.output, exist_ok=True)
    
    # Carica i risultati
    print(f"Caricamento risultati da {args.input}...")
    df = load_results_with_environment(args.input)
    
    if df is None or df.empty:
        print("Nessun dato trovato. Verifica il percorso dei file CSV.")
        return
    
    print(f"Dati caricati: {len(df)} righe")
    print(f"Colonne disponibili: {list(df.columns)}")
    
    # Calcola metriche goodput
    print("Calcolando metriche goodput...")
    goodput_df = calculate_goodput_metrics(df)
    
    if goodput_df.empty:
        print("Impossibile calcolare metriche goodput. Verificare i dati.")
        return
    
    print(f"Configurazioni goodput: {len(goodput_df)}")
    print(f"Esecuzioni uniche: {goodput_df['run_number'].nunique()}")
    
    # Imposta stile grafici
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Genera analisi
    print("Generazione analisi goodput...")
    plot_goodput_analysis(goodput_df, args.output)
    plot_goodput_comparison(goodput_df, args.output)
    generate_goodput_report(goodput_df, args.output)
    
    print(f"\nAnalisi goodput completata. File salvati in: {args.output}")
    print("\nFile generati:")
    for file in os.listdir(args.output):
        if file.endswith(('.png', '.pdf', '.svg', '.txt')):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
