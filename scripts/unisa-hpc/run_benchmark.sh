#!/bin/bash
# run_benchmark.sh - Esegue i benchmark per una collettiva, libreria e dtype specifici
#
# Usage: ./run_benchmark.sh <library> <collective> <dtype>
# Esempio: ./run_benchmark.sh nccl allreduce float
# Multi-nodo: PPN=4 NUM_GPUS=8 ./run_benchmark.sh oneccl-nvidia allreduce float
#
# NOTA: Eseguire all'interno di una sessione srun interattiva, dopo aver fatto:
#   source /home/S.SIRICA3/scripts/export_variables.sh
#
# Variabili di ambiente opzionali:
#   NUM_GPUS     - Numero di GPU/rank MPI (default: 4)
#   NUM_ITERS    - Numero di iterazioni per ogni size (default: 5)
#   PPN          - Processi per nodo per multi-nodo (se impostato, aggiunge -ppn a mpirun)
#   PROJECT_ROOT - Root del progetto (default: /home/S.SIRICA3/multi-gpu-collectives)

set -euo pipefail

# ===================== Parametri =====================
if [ $# -lt 3 ]; then
    echo "Usage: $0 <library> <collective> <dtype>"
    echo ""
    echo "  library:    nccl | oneccl | oneccl-nvidia"
    echo "  collective: allreduce | alltoall"
    echo "  dtype:      int | float | double"
    echo ""
    echo "Variabili di ambiente opzionali:"
    echo "  NUM_GPUS=4       Numero di GPU (default: 4)"
    echo "  NUM_ITERS=5      Iterazioni per size (default: 5)"
    echo "  PPN=4            Processi per nodo (multi-nodo)"
    echo "  PROJECT_ROOT=... Root del progetto"
    exit 1
fi

LIBRARY="$1"
COLLECTIVE="$2"
DTYPE="$3"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ITERS="${NUM_ITERS:-5}"
PPN="${PPN:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/S.SIRICA3/multi-gpu-collectives}"

# ===================== Validazione =====================
if [[ "$LIBRARY" != "nccl" && "$LIBRARY" != "oneccl" && "$LIBRARY" != "oneccl-nvidia" ]]; then
    echo "Errore: library deve essere 'nccl', 'oneccl' o 'oneccl-nvidia', ricevuto: '$LIBRARY'"
    exit 1
fi

if [[ "$COLLECTIVE" != "allreduce" && "$COLLECTIVE" != "alltoall" ]]; then
    echo "Errore: collective deve essere 'allreduce' o 'alltoall', ricevuto: '$COLLECTIVE'"
    exit 1
fi

case "$DTYPE" in
    int|float) ELEM_SIZE=4 ;;
    double)    ELEM_SIZE=8 ;;
    *)
        echo "Errore: dtype deve essere 'int', 'float' o 'double', ricevuto: '$DTYPE'"
        exit 1
        ;;
esac

BINARY="${PROJECT_ROOT}/build/${LIBRARY}/${COLLECTIVE}"
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

# ===================== Minimo count per collettiva =====================
# allreduce: local_count = count / num_ranks  -> count >= num_ranks
# alltoall:  count_per_dest = count / num_ranks^2  -> count >= num_ranks^2
if [ "$COLLECTIVE" = "alltoall" ]; then
    MIN_COUNT=$((NUM_GPUS * NUM_GPUS))
else
    MIN_COUNT=$NUM_GPUS
fi

# ===================== Target message sizes =====================
# (bytes, label)
SIZES_BYTES=(1 8 64 512 4096 32768 262144 2097152 16777216 134217728 1073741824)
SIZES_LABEL=("1B" "8B" "64B" "512B" "4KiB" "32KiB" "256KiB" "2MiB" "16MiB" "128MiB" "1GiB")

# ===================== Header =====================
echo "============================================================"
echo " Benchmark: ${LIBRARY} | ${COLLECTIVE} | ${DTYPE}"
echo " Ranks: ${NUM_GPUS} | PPN: ${PPN:-N/A} | Iterazioni: ${NUM_ITERS}"
echo " mpirun: mpirun ${MPIRUN_ARGS} ${BINARY} ..."
echo " Output:  ${OUTPUT_DIR}"
echo "============================================================"
echo ""

SKIPPED=0
EXECUTED=0
FAILED=0

for i in "${!SIZES_BYTES[@]}"; do
    MSG_SIZE=${SIZES_BYTES[$i]}
    LABEL=${SIZES_LABEL[$i]}

    # Calcolo count (numero elementi) = msg_size_bytes / sizeof(dtype)
    COUNT=$((MSG_SIZE / ELEM_SIZE))

    # Verifica che il count sia sufficiente per la collettiva
    if [ "$COUNT" -lt "$MIN_COUNT" ]; then
        echo "[SKIP] ${LABEL} (${MSG_SIZE} bytes) -> count=${COUNT}, minimo richiesto=${MIN_COUNT} per ${COLLECTIVE} con ${NUM_GPUS} GPU"
        ((SKIPPED++)) || true
        continue
    fi

    echo "--- ${LABEL} | count=${COUNT} | ${NUM_ITERS} iterazioni ---"

    for iter in $(seq 1 "$NUM_ITERS"); do
        echo -n "  [${iter}/${NUM_ITERS}] "
        if ! mpirun $MPIRUN_ARGS "$BINARY" --dtype "$DTYPE" --count "$COUNT" --output "$OUTPUT_DIR" 2>&1; then
            echo "  [ERRORE] Iterazione ${iter} per ${LABEL} fallita"
            ((FAILED++)) || true
        fi
    done

    ((EXECUTED++)) || true
    echo ""
done

# ===================== Riepilogo =====================
echo "============================================================"
echo " Riepilogo: ${LIBRARY} | ${COLLECTIVE} | ${DTYPE}"
echo "   Size eseguite: ${EXECUTED}"
echo "   Size skippate: ${SKIPPED}"
echo "   Errori:        ${FAILED}"
echo " Risultati in: ${OUTPUT_DIR}"
echo "============================================================"
