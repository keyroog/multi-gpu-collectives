#!/bin/bash
# rerun_specific.sh - Riesegue le 5 iterazioni per un sottoinsieme specifico di
#                     (library, collective, dtype, count) con outlier elevati.
#
# Usage: ./rerun_specific.sh <library> <collective> <dtype> <count1> [count2 ...]
# Esempio (4 rank, intra-nodo):
#   ./rerun_specific.sh nccl allreduce float 1024 524288
# Esempio (8 rank, inter-nodo):
#   PPN=4 NUM_GPUS=8 ./rerun_specific.sh nccl allreduce int 65536 1024 16
#
# NOTA: Eseguire all'interno di una sessione srun interattiva, dopo aver fatto:
#   source /home/S.SIRICA3/scripts/export_variables.sh
#
# <count> è il numero di ELEMENTI (corrisponde al suffisso numerico nei file CSV,
#         es. nccl_allreduce_float_1024_results.csv -> count=1024)
#
# Variabili di ambiente opzionali (stesse di run_benchmark.sh):
#   NUM_GPUS     - Numero di GPU/rank MPI (default: 4)
#   NUM_ITERS    - Numero di iterazioni per ogni size (default: 5)
#   PPN          - Processi per nodo per multi-nodo (se impostato, aggiunge -ppn a mpirun)
#   PROJECT_ROOT - Root del progetto (default: /home/S.SIRICA3/multi-gpu-collectives)
#   OVERWRITE    - Se "1", cancella le righe esistenti prima di appendere (default: 0)

set -euo pipefail

# ===================== Parametri =====================
if [ $# -lt 4 ]; then
    echo "Usage: $0 <library> <collective> <dtype> <count1> [count2 ...]"
    echo ""
    echo "  library:    nccl | oneccl | oneccl-nvidia"
    echo "  collective: allreduce | alltoall"
    echo "  dtype:      int | float | double"
    echo "  count:      numero elementi (es. 1024, 65536, 524288)"
    echo ""
    echo "Variabili di ambiente opzionali:"
    echo "  NUM_GPUS=4       Numero di GPU (default: 4)"
    echo "  NUM_ITERS=5      Iterazioni per size (default: 5)"
    echo "  PPN=4            Processi per nodo (multi-nodo)"
    echo "  PROJECT_ROOT=... Root del progetto"
    echo "  OVERWRITE=1      Cancella il CSV esistente prima di riscrivere"
    echo ""
    echo "Esempio - re-run 8-rank nccl allreduce int per le size più critiche:"
    echo "  PPN=4 NUM_GPUS=8 $0 nccl allreduce int 65536 1024 16 524288 262144"
    exit 1
fi

LIBRARY="$1"
COLLECTIVE="$2"
DTYPE="$3"
shift 3
COUNTS=("$@")   # tutti i count rimanenti

NUM_GPUS="${NUM_GPUS:-4}"
NUM_ITERS="${NUM_ITERS:-5}"
PPN="${PPN:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/S.SIRICA3/multi-gpu-collectives}"
OVERWRITE="${OVERWRITE:-0}"

# ===================== Validazione libreria / collettiva / dtype =====================
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

# Binario: oneccl-nvidia usa il binario oneccl
BIN_LIBRARY="$LIBRARY"
if [ "$LIBRARY" = "oneccl-nvidia" ]; then
    BIN_LIBRARY="oneccl"
fi

BINARY="${PROJECT_ROOT}/build/${BIN_LIBRARY}/${COLLECTIVE}"
if [ ! -x "$BINARY" ]; then
    echo "Errore: binario non trovato o non eseguibile: $BINARY"
    echo "Assicurati di aver compilato con: make ${BIN_LIBRARY}"
    exit 1
fi

# ===================== Minimum count check =====================
if [ "$COLLECTIVE" = "alltoall" ]; then
    MIN_COUNT=$((NUM_GPUS * NUM_GPUS))
else
    MIN_COUNT=$NUM_GPUS
fi

# ===================== Costruzione comando mpirun =====================
MPIRUN_ARGS="-np $NUM_GPUS"
if [ -n "$PPN" ]; then
    MPIRUN_ARGS="-ppn $PPN $MPIRUN_ARGS"
fi

# ===================== Output directory =====================
OUTPUT_DIR="${PROJECT_ROOT}/results/unisa-hpc/${NUM_GPUS}_rank/${LIBRARY}"
mkdir -p "$OUTPUT_DIR"

# ===================== Header =====================
echo "============================================================"
echo " RE-RUN specifico: ${LIBRARY} | ${COLLECTIVE} | ${DTYPE}"
echo " Ranks: ${NUM_GPUS} | PPN: ${PPN:-N/A} | Iterazioni: ${NUM_ITERS}"
echo " mpirun: mpirun ${MPIRUN_ARGS} ${BINARY} ..."
echo " Output:  ${OUTPUT_DIR}"
echo " Sizes da rieseguire: ${COUNTS[*]}"
echo " OVERWRITE=${OVERWRITE}"
echo "============================================================"
echo ""

EXECUTED=0
FAILED=0
SKIPPED=0

for COUNT in "${COUNTS[@]}"; do
    # Validazione: il count deve essere un intero positivo
    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        echo "[SKIP] count='${COUNT}' non è un intero valido, saltato."
        ((SKIPPED++)) || true
        continue
    fi

    if [ "$COUNT" -lt "$MIN_COUNT" ]; then
        echo "[SKIP] count=${COUNT} inferiore al minimo richiesto=${MIN_COUNT} per ${COLLECTIVE} con ${NUM_GPUS} GPU"
        ((SKIPPED++)) || true
        continue
    fi

    # Dimensione in byte (solo per log)
    BYTES=$((COUNT * ELEM_SIZE))

    # Percorso CSV che verrà scritto dal binario
    # Naming convention: {library}_{collective}_{dtype}_{count}_results.csv
    CSV_FILE="${OUTPUT_DIR}/${LIBRARY}_${COLLECTIVE}_${DTYPE}_${COUNT}_results.csv"

    if [ "$OVERWRITE" = "1" ] && [ -f "$CSV_FILE" ]; then
        echo "[OVERWRITE] Rimozione ${CSV_FILE}"
        rm -f "$CSV_FILE"
    fi

    echo "--- count=${COUNT} (${BYTES} bytes) | ${NUM_ITERS} iterazioni ---"

    for iter in $(seq 1 "$NUM_ITERS"); do
        echo -n "  [${iter}/${NUM_ITERS}] "
        if ! mpirun $MPIRUN_ARGS "$BINARY" --dtype "$DTYPE" --count "$COUNT" --output "$OUTPUT_DIR" 2>&1; then
            echo "  [ERRORE] Iterazione ${iter} per count=${COUNT} fallita"
            ((FAILED++)) || true
        fi
    done

    ((EXECUTED++)) || true
    echo ""
done

# ===================== Riepilogo =====================
echo "============================================================"
echo " Riepilogo re-run: ${LIBRARY} | ${COLLECTIVE} | ${DTYPE}"
echo "   Size eseguite: ${EXECUTED}"
echo "   Size skippate: ${SKIPPED}"
echo "   Errori:        ${FAILED}"
echo " Risultati in: ${OUTPUT_DIR}"
echo "============================================================"
