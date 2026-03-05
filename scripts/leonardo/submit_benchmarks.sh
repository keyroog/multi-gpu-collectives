#!/bin/bash
# Usage: ./submit_benchmarks.sh [dtype] [config]
#   dtype:  float | int | double  (default: float)
#   config: 4r | 8r | all         (default: all)

DTYPE="${1:-float}"
CONFIG="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIBRARIES=(nccl oneccl mpi)
COLLECTIVES=(allreduce alltoall)

submit() {
    local sbatch_file="$1"
    local label="$2"

    for LIB in "${LIBRARIES[@]}"; do
        for COLL in "${COLLECTIVES[@]}"; do
            JOB_NAME="${LIB}_${COLL}_${DTYPE}_${label}"
            JOB_ID=$(sbatch --job-name="${JOB_NAME}" "${sbatch_file}" "${LIB}" "${COLL}" "${DTYPE}" | awk '{print $NF}')
            echo "  ${JOB_NAME} -> ${JOB_ID}"
        done
    done
}

case "$CONFIG" in
    4r)  submit "${SCRIPT_DIR}/run_benchmark_4r.sbatch" "4r" ;;
    8r)  submit "${SCRIPT_DIR}/run_benchmark_8r.sbatch" "8r" ;;
    all)
        submit "${SCRIPT_DIR}/run_benchmark_4r.sbatch" "4r"
        submit "${SCRIPT_DIR}/run_benchmark_8r.sbatch" "8r"
        ;;
    *)   echo "config: 4r | 8r | all"; exit 1 ;;
esac

echo "squeue -u \$USER"