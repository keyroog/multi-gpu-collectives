#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.err
#SBATCH --exclusive

# run_benchmark.sbatch - Esegue il sweep completo di size per una collettiva
#
# Usage:
#   sbatch --job-name=nccl_allreduce_float run_benchmark.sbatch nccl allreduce float
#   sbatch --job-name=oneccl_alltoall_float run_benchmark.sbatch oneccl allreduce float
#
# Argomenti posizionali:
#   $1  library:    nccl | oneccl
#   $2  collective: allreduce | alltoall
#   $3  dtype:      int | float | double

set -euo pipefail

# ===================== Parametri =====================
if [ $# -lt 3 ]; then
    echo "Usage: sbatch run_benchmark.sbatch <library> <collective> <dtype>"
    echo "  library:    nccl | oneccl"
    echo "  collective: allreduce | alltoall"
    echo "  dtype:      int | float | double"
    exit 1
fi

LIBRARY="$1"
COLLECTIVE="$2"
DTYPE="$3"
NUM_GPUS="${SLURM_NTASKS:-4}"
NUM_ITERS="${NUM_ITERS:-5}"
PROJECT_ROOT="/leonardo/home/userexternal/ssirica0/multi-gpu-collectives"

# ===================== Validazione =====================
if [[ "$LIBRARY" != "nccl" && "$LIBRARY" != "oneccl" && "$LIBRARY" != "mpi" ]]; then
    echo "Errore: library deve essere 'nccl', 'oneccl' o 'mpi', ricevuto: '$LIBRARY'"
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
    exit 1
fi

# ===================== Ambiente =====================
source /leonardo/home/userexternal/ssirica0/scripts/cuda_export_variables.sh

if [ "$LIBRARY" = "oneccl" ]; then
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables_multinode.sh
    else
        source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables.sh
    fi
fi

# ===================== Output directory =====================
OUTPUT_DIR="${PROJECT_ROOT}/results/leonardo/${NUM_GPUS}_rank/${LIBRARY}"
mkdir -p "$OUTPUT_DIR"

# ===================== Minimo count per collettiva =====================
if [ "$COLLECTIVE" = "alltoall" ]; then
    MIN_COUNT=$((NUM_GPUS * NUM_GPUS))
else
    MIN_COUNT=$NUM_GPUS
fi

# ===================== Target message sizes =====================
SIZES_BYTES=(1 8 64 512 4096 32768 262144 2097152 16777216 134217728 1073741824)
SIZES_LABEL=("1B" "8B" "64B" "512B" "4KiB" "32KiB" "256KiB" "2MiB" "16MiB" "128MiB" "1GiB")

# ===================== Header =====================
echo "============================================================"
echo " Cluster:    Leonardo (CINECA)"
echo " Benchmark:  ${LIBRARY} | ${COLLECTIVE} | ${DTYPE}"
echo " Ranks:      ${NUM_GPUS} | Iterazioni: ${NUM_ITERS}"
echo " Nodo:       $(hostname)"
echo " Job ID:     ${SLURM_JOB_ID}"
echo " Output:     ${OUTPUT_DIR}"
echo "============================================================"
echo ""

SKIPPED=0
EXECUTED=0
FAILED=0

for i in "${!SIZES_BYTES[@]}"; do
    MSG_SIZE=${SIZES_BYTES[$i]}
    LABEL=${SIZES_LABEL[$i]}

    COUNT=$((MSG_SIZE / ELEM_SIZE))

    if [ "$COUNT" -lt "$MIN_COUNT" ]; then
        echo "[SKIP] ${LABEL} (${MSG_SIZE} bytes) -> count=${COUNT}, minimo richiesto=${MIN_COUNT}"
        ((SKIPPED++)) || true
        continue
    fi

    echo "--- ${LABEL} | count=${COUNT} | ${NUM_ITERS} iterazioni ---"

    for iter in $(seq 1 "$NUM_ITERS"); do
        echo -n "  [${iter}/${NUM_ITERS}] "
        if ! srun --ntasks="${NUM_GPUS}" \
                  "${BINARY}" --dtype "${DTYPE}" --count "${COUNT}" --output "${OUTPUT_DIR}" 2>&1; then
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
