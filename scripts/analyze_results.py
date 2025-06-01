#!/usr/bin/env python3
"""
Script semplice per verificare e analizzare rapidamente i risultati CSV.
"""

import pandas as pd
import argparse
import os
import glob

def analyze_results(results_dir):
    """Analizza rapidamente i risultati CSV."""
    
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    if not csv_files:
        print(f"❌ Nessun file CSV trovato in {results_dir}")
        return
    
    print(f"📊 Trovati {len(csv_files)} file di risultati")
    print("=" * 50)
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Retrocompatibilità: aggiungi colonne mancanti se non esistono
            if 'run_number' not in df.columns:
                df['run_number'] = range(1, len(df) + 1)
                print(f"✅ {os.path.basename(file)}: {len(df)} righe (aggiunta colonna run_number)")
            else:
                print(f"✅ {os.path.basename(file)}: {len(df)} righe")
            
            if 'environment' not in df.columns:
                df['environment'] = 'legacy'
            
            all_data.append(df)
        except Exception as e:
            print(f"❌ Errore nel leggere {file}: {e}")
    
    if not all_data:
        return
    
    # Combina tutti i dati
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("\n📈 STATISTICHE GENERALI")
    print("=" * 50)
    print(f"Totale esperimenti: {len(combined_df)}")
    print(f"Libraries: {', '.join(combined_df['library'].unique())}")
    print(f"Collectives: {', '.join(combined_df['collective'].unique())}")
    print(f"Data types: {', '.join(combined_df['data_type'].unique())}")
    print(f"Numero di ranks: {', '.join(map(str, sorted(combined_df['num_ranks'].unique())))}")
    print(f"Range message size: {combined_df['message_size_elements'].min():,} - {combined_df['message_size_elements'].max():,} elementi")
    
    print("\n⏱️  PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Raggruppa per data type e mostra statistiche
    for dtype in sorted(combined_df['data_type'].unique()):
        dtype_data = combined_df[combined_df['data_type'] == dtype]
        
        print(f"\n{dtype.upper()}:")
        print(f"  Tempo medio: {dtype_data['time_ms'].mean():.3f} ± {dtype_data['time_ms'].std():.3f} ms")
        print(f"  Tempo minimo: {dtype_data['time_ms'].min():.3f} ms")
        print(f"  Tempo massimo: {dtype_data['time_ms'].max():.3f} ms")
        print(f"  Numero di test: {len(dtype_data)}")
        
        # Trova il test più veloce
        fastest = dtype_data.loc[dtype_data['time_ms'].idxmin()]
        print(f"  Test più veloce: {fastest['time_ms']:.3f}ms con {fastest['message_size_elements']:,} elementi (rank {fastest['rank']})")
    
    print("\n📊 PERFORMANCE PER MESSAGE SIZE")
    print("=" * 50)
    
    # Raggruppa per message size
    size_stats = combined_df.groupby('message_size_elements')['time_ms'].agg(['mean', 'std', 'count']).round(3)
    
    print(size_stats.to_string())
    
    # Trend analysis
    print("\n📈 TREND ANALYSIS")
    print("=" * 50)
    
    # Calcola il throughput (elementi/ms)
    combined_df['throughput'] = combined_df['message_size_elements'] / combined_df['time_ms']
    
    best_throughput = combined_df.loc[combined_df['throughput'].idxmax()]
    print(f"Miglior throughput: {best_throughput['throughput']:.0f} elementi/ms")
    print(f"  - Data type: {best_throughput['data_type']}")
    print(f"  - Message size: {best_throughput['message_size_elements']:,} elementi")
    print(f"  - Tempo: {best_throughput['time_ms']:.3f} ms")
    print(f"  - Rank: {best_throughput['rank']}")
    
    print("\n🔍 CONSIGLI PER L'ANALISI")
    print("=" * 50)
    print("• Usa lo script generate_plots.py per grafici dettagliati")
    print("• Controlla la consistenza tra i ranks")
    print("• Analizza la scalabilità con message size crescenti")
    print("• Confronta le performance tra data types")
    
    if combined_df['num_ranks'].nunique() > 1:
        print("• Analizza la scalabilità con più ranks")

def main():
    parser = argparse.ArgumentParser(description='Analizza rapidamente i risultati CSV')
    parser.add_argument('--input', required=True, help='Directory contenente i file CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Directory {args.input} non trovata")
        return
    
    analyze_results(args.input)

if __name__ == "__main__":
    main()
