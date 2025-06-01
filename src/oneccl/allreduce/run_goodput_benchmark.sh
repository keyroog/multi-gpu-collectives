#!/bin/bash

# Script avanzato per benchmark OneCCL AllReduce con analisi goodput
# Implementa multiple esecuzioni e analisi worst-rank timing

set -e

# Configurazione
EXECUTABLE="./allreduce"
OUTPUT_DIR="../../../results/oneccl"
NUM_RANKS=2
NUM_RUNS=5  # Numero di esecuzioni complete per ogni test

# Data types da testare
DATA_TYPES=("int" "float" "double")

# Message sizes da testare (in numero di elementi)
MESSAGE_SIZES=(1024 4096 16384 65536 262144 1048576)

# Environment variables per debugging performance
export CCL_LOG_LEVEL=warn  # Cambia a 'trace' per debug dettagliato
export ZE_DEBUG=0          # Cambia a 1 per debug Level-Zero
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1  # Ordinamento consistente device
export OMP_NUM_THREADS=1   # Controllo threading

# Crea directory di output
mkdir -p "$OUTPUT_DIR"

echo "=== OneCCL AllReduce Goodput Benchmark ==="
echo "Output directory: $OUTPUT_DIR"
echo "Number of ranks: $NUM_RANKS"
echo "Number of runs per test: $NUM_RUNS"
echo "Data types: ${DATA_TYPES[*]}"
echo "Message sizes: ${MESSAGE_SIZES[*]}"
echo
echo "Environment settings:"
echo "  CCL_LOG_LEVEL=$CCL_LOG_LEVEL"
echo "  ZE_DEBUG=$ZE_DEBUG"
echo "  ZE_ENABLE_PCI_ID_DEVICE_ORDER=$ZE_ENABLE_PCI_ID_DEVICE_ORDER"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo

# Verifica che l'eseguibile esista
if [ ! -x "$EXECUTABLE" ]; then
    echo "Errore: $EXECUTABLE non trovato o non eseguibile"
    echo "Compila prima il programma con: icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl"
    exit 1
fi

# Function per eseguire un singolo test
run_single_test() {
    local dtype=$1
    local size=$2
    local run_num=$3
    local total_runs=$4
    
    echo "  Run $run_num/$total_runs: $dtype with $size elements..."
    
    # File temporaneo per catturare l'output
    local temp_output="/tmp/benchmark_run_${dtype}_${size}_${run_num}.log"
    
    # Esegui il test e cattura l'output
    if mpirun -n $NUM_RANKS $EXECUTABLE \
        --dtype "$dtype" \
        --count "$size" \
        --output "$OUTPUT_DIR" > "$temp_output" 2>&1; then
        
        # Estrai le informazioni di timing dal log
        local rank0_time=$(grep "Rank 0 allreduce time" "$temp_output" | grep -o '[0-9]*\.[0-9]*' | head -1)
        local rank1_time=$(grep "Rank 1 allreduce time" "$temp_output" | grep -o '[0-9]*\.[0-9]*' | head -1)
        
        if [[ -n "$rank0_time" && -n "$rank1_time" ]]; then
            # Calcola goodput (worst-rank time)
            local goodput_time=$(echo "$rank0_time $rank1_time" | awk '{print ($1 > $2) ? $1 : $2}')
            echo "    Goodput (worst-rank): ${goodput_time}ms (Rank0: ${rank0_time}ms, Rank1: ${rank1_time}ms)"
        fi
        
        echo "    ✓ Completed successfully"
    else
        echo "    ✗ Failed - check $temp_output for details"
        cat "$temp_output"
    fi
    
    # Cleanup temporary file
    rm -f "$temp_output"
    
    # Breve pausa tra le esecuzioni
    sleep 1
}

# Esegui i benchmark
total_tests=$((${#DATA_TYPES[@]} * ${#MESSAGE_SIZES[@]} * NUM_RUNS))
current_test=0

for dtype in "${DATA_TYPES[@]}"; do
    echo "Testing data type: $dtype"
    echo "----------------------------------------"
    
    for size in "${MESSAGE_SIZES[@]}"; do
        echo "Message size: $size elements"
        
        # Array per raccogliere i tempi di tutte le esecuzioni
        declare -a all_times=()
        
        for run in $(seq 1 $NUM_RUNS); do
            current_test=$((current_test + 1))
            echo "[$current_test/$total_tests]"
            
            run_single_test "$dtype" "$size" "$run" "$NUM_RUNS"
        done
        
        echo "  Completed all runs for $dtype/$size"
        echo
    done
    echo "Completed all tests for $dtype"
    echo "========================================"
    echo
done

echo "=== Benchmark completato ==="
echo "Risultati salvati in: $OUTPUT_DIR"
echo
echo "Per generare i grafici con analisi goodput:"
echo "python3 ../../../scripts/generate_goodput_plots.py --input $OUTPUT_DIR --output ../../../results/plots"
echo

# Mostra una anteprima dei risultati
echo "=== ANTEPRIMA RISULTATI ==="
for file in "$OUTPUT_DIR"/*_results.csv; do
    if [[ -f "$file" ]]; then
        echo "File: $(basename "$file")"
        echo "  Righe: $(wc -l < "$file")"
        echo "  Ultima esecuzione: $(tail -1 "$file" | cut -d',' -f9)"
    fi
done

echo
echo "=== CONSIGLI PER L'ANALISI ==="
echo "1. Analizza la consistenza dei tempi tra le esecuzioni"
echo "2. Confronta goodput (worst-rank) vs average timing"
echo "3. Verifica se ci sono variazioni significative negli environment settings"
echo "4. Per debugging dettagliato, riavvia con CCL_LOG_LEVEL=trace"
echo "5. Usa 'ocloc query' per informazioni dettagliate sulle GPU Intel Max"
