#!/usr/bin/env python3
"""
Script per generare grafici dai risultati dei benchmark delle collective operations.
Legge i file CSV generati dal logger e crea grafici per analizzare le performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from pathlib import Path

def load_results(results_dir):
    """Carica tutti i file CSV dalla directory dei risultati."""
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {results_dir}")
        return None
    
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Caricato: {file} ({len(df)} righe)")
        except Exception as e:
            print(f"Errore nel caricare {file}: {e}")
    
    if not dataframes:
        return None
    
    # Combina tutti i dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def plot_performance_by_message_size(df, output_dir):
    """Crea grafici delle performance in funzione della dimensione del messaggio."""
    
    # Raggruppa per library, collective, data_type e message_size
    grouped = df.groupby(['library', 'collective', 'data_type', 'message_size_elements'])['time_ms'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Crea subplot per ogni combinazione library/collective
    libraries = df['library'].unique()
    collectives = df['collective'].unique()
    data_types = df['data_type'].unique()
    
    for lib in libraries:
        for collective in collectives:
            lib_collective_data = grouped[(grouped['library'] == lib) & (grouped['collective'] == collective)]
            
            if lib_collective_data.empty:
                continue
            
            fig, axes = plt.subplots(1, len(data_types), figsize=(6*len(data_types), 6))
            if len(data_types) == 1:
                axes = [axes]
            
            fig.suptitle(f'{lib.upper()} - {collective.capitalize()} Performance vs Message Size', fontsize=16)
            
            for i, dtype in enumerate(data_types):
                dtype_data = lib_collective_data[lib_collective_data['data_type'] == dtype]
                
                if dtype_data.empty:
                    axes[i].text(0.5, 0.5, f'No data for {dtype}', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'Data Type: {dtype}')
                    continue
                
                # Ordina per message size
                dtype_data = dtype_data.sort_values('message_size_elements')
                
                # Plot con error bars
                axes[i].errorbar(dtype_data['message_size_elements'], dtype_data['mean'], 
                               yerr=dtype_data['std'], marker='o', capsize=5, capthick=2, linewidth=2)
                
                axes[i].set_xlabel('Message Size (elements)')
                axes[i].set_ylabel('Time (ms)')
                axes[i].set_title(f'Data Type: {dtype}')
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=0.3)
                
                # Aggiungi annotazioni per min/max
                for _, row in dtype_data.iterrows():
                    axes[i].annotate(f'min: {row["min"]:.2f}ms\nmax: {row["max"]:.2f}ms', 
                                   xy=(row['message_size_elements'], row['mean']),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            
            # Salva il grafico
            filename = f'{lib}_{collective}_performance_vs_message_size.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Grafico salvato: {filepath}")
            plt.close()

def plot_performance_by_data_type(df, output_dir):
    """Crea grafici per confrontare le performance tra diversi data types."""
    
    # Raggruppa per library, collective, data_type
    grouped = df.groupby(['library', 'collective', 'data_type', 'message_size_elements'])['time_ms'].mean().reset_index()
    
    libraries = df['library'].unique()
    collectives = df['collective'].unique()
    
    for lib in libraries:
        for collective in collectives:
            lib_collective_data = grouped[(grouped['library'] == lib) & (grouped['collective'] == collective)]
            
            if lib_collective_data.empty:
                continue
            
            plt.figure(figsize=(12, 8))
            
            # Pivot per avere data_type come colonne
            pivot_data = lib_collective_data.pivot_table(
                index='message_size_elements', 
                columns='data_type', 
                values='time_ms'
            )
            
            # Plot per ogni data type
            for dtype in pivot_data.columns:
                plt.plot(pivot_data.index, pivot_data[dtype], marker='o', label=dtype, linewidth=2)
            
            plt.xlabel('Message Size (elements)')
            plt.ylabel('Time (ms)')
            plt.title(f'{lib.upper()} - {collective.capitalize()}: Performance by Data Type')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Salva il grafico
            filename = f'{lib}_{collective}_performance_by_datatype.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Grafico salvato: {filepath}")
            plt.close()

def plot_scaling_analysis(df, output_dir):
    """Analizza la scalabilità in funzione del numero di ranks."""
    
    if 'num_ranks' not in df.columns or df['num_ranks'].nunique() <= 1:
        print("Dati insufficienti per l'analisi di scalabilità (serve più di un numero di ranks)")
        return
    
    # Raggruppa per tutti i parametri rilevanti
    grouped = df.groupby(['library', 'collective', 'data_type', 'message_size_elements', 'num_ranks'])['time_ms'].mean().reset_index()
    
    libraries = df['library'].unique()
    collectives = df['collective'].unique()
    data_types = df['data_type'].unique()
    
    for lib in libraries:
        for collective in collectives:
            for dtype in data_types:
                subset = grouped[
                    (grouped['library'] == lib) & 
                    (grouped['collective'] == collective) & 
                    (grouped['data_type'] == dtype)
                ]
                
                if subset.empty:
                    continue
                
                plt.figure(figsize=(10, 6))
                
                # Plot per diverse message sizes
                message_sizes = subset['message_size_elements'].unique()
                for msg_size in sorted(message_sizes):
                    size_data = subset[subset['message_size_elements'] == msg_size]
                    if len(size_data) > 1:  # Solo se abbiamo più punti
                        plt.plot(size_data['num_ranks'], size_data['time_ms'], 
                               marker='o', label=f'{msg_size} elements', linewidth=2)
                
                plt.xlabel('Number of Ranks')
                plt.ylabel('Time (ms)')
                plt.title(f'{lib.upper()} - {collective.capitalize()} ({dtype}): Scaling Analysis')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                # Salva il grafico
                filename = f'{lib}_{collective}_{dtype}_scaling_analysis.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Grafico salvato: {filepath}")
                plt.close()

def generate_summary_report(df, output_dir):
    """Genera un report di riepilogo."""
    
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== BENCHMARK SUMMARY REPORT ===\n\n")
        
        f.write(f"Total number of experiments: {len(df)}\n")
        f.write(f"Libraries tested: {', '.join(df['library'].unique())}\n")
        f.write(f"Collectives tested: {', '.join(df['collective'].unique())}\n")
        f.write(f"Data types tested: {', '.join(df['data_type'].unique())}\n")
        f.write(f"Message sizes range: {df['message_size_elements'].min()} - {df['message_size_elements'].max()} elements\n")
        f.write(f"Number of ranks tested: {', '.join(map(str, sorted(df['num_ranks'].unique())))}\n\n")
        
        # Performance summary per combinazione
        f.write("=== PERFORMANCE SUMMARY ===\n")
        summary = df.groupby(['library', 'collective', 'data_type'])['time_ms'].agg(['mean', 'std', 'min', 'max']).round(3)
        f.write(summary.to_string())
        f.write("\n\n")
        
        # Best performance per data type
        f.write("=== BEST PERFORMANCE BY DATA TYPE ===\n")
        for dtype in df['data_type'].unique():
            dtype_data = df[df['data_type'] == dtype]
            best_row = dtype_data.loc[dtype_data['time_ms'].idxmin()]
            f.write(f"{dtype}: {best_row['time_ms']:.3f}ms ({best_row['library']} {best_row['collective']}, "
                   f"{best_row['message_size_elements']} elements, {best_row['num_ranks']} ranks)\n")
    
    print(f"Report di riepilogo salvato: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Genera grafici dai risultati dei benchmark')
    parser.add_argument('--input', required=True, help='Directory contenente i file CSV dei risultati')
    parser.add_argument('--output', required=True, help='Directory dove salvare i grafici')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], 
                       help='Formato dei grafici (default: png)')
    
    args = parser.parse_args()
    
    # Crea directory di output se non esiste
    os.makedirs(args.output, exist_ok=True)
    
    # Carica i risultati
    print(f"Caricamento risultati da {args.input}...")
    df = load_results(args.input)
    
    if df is None or df.empty:
        print("Nessun dato trovato. Verifica il percorso dei file CSV.")
        return
    
    print(f"Dati caricati: {len(df)} righe")
    print(f"Colonne disponibili: {list(df.columns)}")
    
    # Imposta lo stile dei grafici
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Genera i grafici
    print("Generazione grafici...")
    plot_performance_by_message_size(df, args.output)
    plot_performance_by_data_type(df, args.output)
    plot_scaling_analysis(df, args.output)
    
    # Genera report di riepilogo
    generate_summary_report(df, args.output)
    
    print(f"Tutti i grafici sono stati salvati in {args.output}")

if __name__ == "__main__":
    main()
