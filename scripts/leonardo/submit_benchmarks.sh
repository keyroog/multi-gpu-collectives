#!/bin/bash
# submit_benchmarks.sh - Sottomette tutti i benchmark nccl/oneccl allreduce/alltoall
#
# Usage: ./submit_benchmarks.sh [dtype] [config]
#   dtype:  float | int | double  (default: float)
#   config: 4r | 8r | all         (default: all)
#
# Configurazioni:
#   4r  -> 4 rank, 1 nodo,  4 GPU
#   8r  -> 8 rank, 2 nodi,  4 GPU/nodo
#   all -> entrambe

DTYPE="${1:-float}"
CONFIG="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_benchmark.sbatch"

LIBRARIES=(nccl oneccl mpi)
COLLECTIVES=(allreduce alltoall)

submit_config() {
    local nodes="$1"
    local ntasks="$2"
    local label="$3"

    echo "--- Configurazione: ${label} (${nodes} nodo/i, ${ntasks} rank) ---"
    for LIB in "${LIBRARIES[@]}"; do
        for COLL in "${COLLECTIVES[@]}"; do
            JOB_NAME="${LIB}_${COLL}_${DTYPE}_${label}"
            JOB_ID=$(sbatch \
                --nodes="${nodes}" \
                --ntasks="${ntasks}" \
                --gres=gpu:4 \
                --job-name="${JOB_NAME}" \
                "${SBATCH_SCRIPT}" "${LIB}" "${COLL}" "${DTYPE}" \
                | awk '{print $NF}')
            echo "  Submitted ${JOB_NAME} -> job ${JOB_ID}"
        done
    done
    echo ""
}

echo "Sottomissione benchmark su Leonardo | dtype=${DTYPE} | config=${CONFIG}"
echo ""

case "$CONFIG" in
    4r)  submit_config 1 4 "4r" ;;
    8r)  submit_config 2 8 "8r" ;;
    all)
        submit_config 1 4 "4r"
        submit_config 2 8 "8r"
        ;;
    *)
        echo "Errore: config deve essere '4r', '8r' o 'all', ricevuto: '$CONFIG'"
        exit 1
        ;;
esac

echo "Monitora con: squeue -u \$USER"
