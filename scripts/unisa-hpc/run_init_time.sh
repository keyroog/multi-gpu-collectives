#!/bin/bash
# run_init_time.sh - Esegue il benchmark di init time per una libreria
#
# Usage: ./run_init_time.sh <library>
# Esempio: ./run_init_time.sh nccl
# Multi-nodo: PPN=4 NUM_GPUS=8 ./run_init_time.sh oneccl
#
# NOTA: Eseguire all'interno di una sessione srun interattiva, dopo aver fatto:
#   source /home/S.SIRICA3/scripts/export_variables.sh
#
# Variabili di ambiente opzionali:
#   NUM_GPUS     - Numero di GPU/rank MPI (default: 4)
#   NUM_ITERS    - Numero di esecuzioni mpirun indipendenti (default: 10)
#   PPN          - Processi per nodo per multi-nodo (se impostato, aggiunge -ppn a mpirun)
#   PROJECT_ROOT - Root del progetto (default: /home/S.SIRICA3/multi-gpu-collectives)

set -euo pipefail

# ===================== Parametri =====================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <library>"
    echo ""
    echo "  library: nccl | oneccl | rccl"
    echo ""
    echo "Variabili di ambiente opzionali:"
    echo "  NUM_GPUS=4       Numero di GPU (default: 4)"
    echo "  NUM_ITERS=10     Numero di esecuzioni indipendenti (default: 10)"
    echo "  PPN=4            Processi per nodo (multi-nodo)"
    echo "  PROJECT_ROOT=... Root del progetto"
    exit 1
fi

LIBRARY="$1"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ITERS="${NUM_ITERS:-10}"
PPN="${PPN:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/S.SIRICA3/multi-gpu-collectives}"

# ===================== Validazione =====================
if [[ "$LIBRARY" != "nccl" && "$LIBRARY" != "oneccl" && "$LIBRARY" != "rccl" ]]; then
    echo "Errore: library deve essere 'nccl', 'oneccl' o 'rccl', ricevuto: '$LIBRARY'"
    exit 1
fi

BINARY="${PROJECT_ROOT}/build/${LIBRARY}/init_time"
if [ ! -x "$BINARY" ]; then
    echo "Errore: binario non trovato o non eseguibile: $BINARY"
    echo "Assicurati di aver compilato con: make ${LIBRARY}"
    exit 1
fi

# ===================== Costruzione comando mpirun =====================
MPIRUN_ARGS="-np $NUM_GPUS"
if [ -n "$PPN" ]; then
    MPIRUN_ARGS="-ppn $PPN $MPIRUN_ARGS"
fi

# ===================== Output directory =====================
OUTPUT_DIR="${PROJECT_ROOT}/results/unisa-hpc/${NUM_GPUS}_rank/${LIBRARY}"
mkdir -p "$OUTPUT_DIR"

# Rimuovi CSV esistente: ogni lancio dello script ricomincia da zero
CSV_FILE="${OUTPUT_DIR}/${LIBRARY}_init_time_${NUM_GPUS}ranks_results.csv"
if [ -f "$CSV_FILE" ]; then
    echo "Rimuovo CSV precedente: $CSV_FILE"
    rm "$CSV_FILE"
fi

# ===================== Header =====================
echo "============================================================"
echo " Init Time Benchmark: ${LIBRARY}"
echo " Ranks: ${NUM_GPUS} | PPN: ${PPN:-N/A} | Esecuzioni: ${NUM_ITERS}"
echo " Binario: ${BINARY}"
echo " Output:  ${OUTPUT_DIR}"
echo "============================================================"
echo ""

FAILED=0

for iter in $(seq 1 "$NUM_ITERS"); do
    echo -n "  Esecuzione [${iter}/${NUM_ITERS}]: "
    if ! mpirun $MPIRUN_ARGS "$BINARY" --iter "$iter" --output "$OUTPUT_DIR" 2>&1; then
        echo "  [ERRORE] Esecuzione ${iter} fallita"
        ((FAILED++)) || true
    fi
done

# ===================== Riepilogo =====================
echo ""
echo "============================================================"
echo " Riepilogo: ${LIBRARY} | ${NUM_GPUS} rank"
echo "   Esecuzioni riuscite: $((NUM_ITERS - FAILED)) / ${NUM_ITERS}"
if [ "$FAILED" -gt 0 ]; then
    echo "   Errori: ${FAILED}"
fi
echo " Risultati in: ${OUTPUT_DIR}"
echo "============================================================"
