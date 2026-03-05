#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.err
#SBATCH --exclusive

# test_single.sbatch - Esegue un singolo test per verifica rapida
#
# Usage:
#   sbatch --job-name=test_nccl test_single.sbatch nccl allreduce float

LIBRARY="$1"
COLLECTIVE="$2"
DTYPE="${3:-float}"
COUNT="${4:-1024}"
PROJECT_ROOT="/leonardo/home/userexternal/ssirica0/multi-gpu-collectives"

source /leonardo/home/userexternal/ssirica0/scripts/cuda_export_variables.sh

if [ "$LIBRARY" = "oneccl" ]; then
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables_multinode.sh
    else
        source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables.sh
    fi
fi

BINARY="${PROJECT_ROOT}/build/${LIBRARY}/${COLLECTIVE}"
OUTPUT_DIR="${PROJECT_ROOT}/results/leonardo/test"
mkdir -p "$OUTPUT_DIR"

echo "Test: ${LIBRARY} | ${COLLECTIVE} | ${DTYPE} | count=${COUNT}"
echo "Nodo: $(hostname) | Job: ${SLURM_JOB_ID}"
echo ""

srun --ntasks="${SLURM_NTASKS}" \
     "${BINARY}" --dtype "${DTYPE}" --count "${COUNT}" --output "${OUTPUT_DIR}"
