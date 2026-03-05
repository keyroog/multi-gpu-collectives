#!/bin/bash
#SBATCH --job-name=bench_8r
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.err
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive

LIBRARY="$1"
COLLECTIVE="$2"
DTYPE="$3"
NUM_GPUS=8
NUM_ITERS="${NUM_ITERS:-5}"
PROJECT_ROOT="/leonardo/home/userexternal/ssirica0/multi-gpu-collectives"
BINARY="${PROJECT_ROOT}/build/${LIBRARY}/${COLLECTIVE}"
OUTPUT_DIR="${PROJECT_ROOT}/results/leonardo/${NUM_GPUS}_rank/${LIBRARY}"

module purge
module load gcc/ cuda/12.2 openmpi/ nvhpc/ nccl/

[ "$LIBRARY" = "oneccl" ] && source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables_multinode.sh

mkdir -p "$OUTPUT_DIR"

case "$DTYPE" in
    int|float) ELEM_SIZE=4 ;;
    double)    ELEM_SIZE=8 ;;
esac

[ "$COLLECTIVE" = "alltoall" ] && MIN_COUNT=$((NUM_GPUS * NUM_GPUS)) || MIN_COUNT=$NUM_GPUS

SIZES_BYTES=(1 8 64 512 4096 32768 262144 2097152 16777216 134217728 1073741824)

for MSG_SIZE in "${SIZES_BYTES[@]}"; do
    COUNT=$((MSG_SIZE / ELEM_SIZE))
    [ "$COUNT" -lt "$MIN_COUNT" ] && continue
    for iter in $(seq 1 "$NUM_ITERS"); do
        srun "${BINARY}" --dtype "${DTYPE}" --count "${COUNT}" --output "${OUTPUT_DIR}"
    done
done