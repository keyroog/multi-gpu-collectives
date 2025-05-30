#!/bin/bash

# Script per eseguire benchmark automatici con OneCCL AllReduce
# Testa diverse combinazioni di data type e message size

set -e

# Configurazione
EXECUTABLE="./allreduce"
OUTPUT_DIR="../results/oneccl"
NUM_RANKS=2

# Data types da testare
DATA_TYPES=("int" "float" "double")

# Message sizes da testare (in numero di elementi)
MESSAGE_SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216)

# Crea directory di output
mkdir -p "$OUTPUT_DIR"

echo "=== OneCCL AllReduce Benchmark ==="
echo "Output directory: $OUTPUT_DIR"
echo "Number of ranks: $NUM_RANKS"
echo "Data types: ${DATA_TYPES[*]}"
echo "Message sizes: ${MESSAGE_SIZES[*]}"
echo

# Verifica che l'eseguibile esista
if [ ! -x "$EXECUTABLE" ]; then
    echo "Errore: $EXECUTABLE non trovato o non eseguibile"
    echo "Compila prima il programma con: icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl"
    exit 1
fi

# Esegui i benchmark
total_tests=$((${#DATA_TYPES[@]} * ${#MESSAGE_SIZES[@]}))
current_test=0

for dtype in "${DATA_TYPES[@]}"; do
    for size in "${MESSAGE_SIZES[@]}"; do
        current_test=$((current_test + 1))
        echo "[$current_test/$total_tests] Testing $dtype with $size elements..."
        
        # Esegui il test
        mpirun -n $NUM_RANKS $EXECUTABLE \
            --dtype "$dtype" \
            --count "$size" \
            --output "$OUTPUT_DIR"
        
        echo "Completed test $current_test/$total_tests"
        echo
        
        # Breve pausa tra i test
        sleep 1
    done
done

echo "=== Benchmark completato ==="
echo "Risultati salvati in: $OUTPUT_DIR"
echo
echo "Per generare i grafici, esegui:"
echo "python3 ../scripts/generate_plots.py --input $OUTPUT_DIR --output ../results/plots"
echo

# Lista i file generati
echo "File generati:"
ls -la "$OUTPUT_DIR"
