#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.err
#SBATCH --exclusive

# run_init_time.sbatch - Benchmark del tempo di inizializzazione per una libreria
#
# Usage:
#   sbatch --job-name=init_nccl_4r run_init_time.sbatch nccl
#   sbatch --nodes=2 --ntasks=8 --job-name=init_oneccl_8r run_init_time.sbatch oneccl
#   sbatch --job-name=init_mpi_4r run_init_time.sbatch mpi
#
# Argomenti posizionali:
#   $1  library: nccl | oneccl | mpi

set -euo pipefail

# ===================== Parametri =====================
if [ $# -lt 1 ]; then
    echo "Usage: sbatch run_init_time.sbatch <library>"
    echo "  library: nccl | oneccl | mpi"
    exit 1
fi

LIBRARY="$1"
NUM_GPUS="${SLURM_NTASKS:-4}"
NUM_ITERS="${NUM_ITERS:-10}"
PROJECT_ROOT="/leonardo/home/userexternal/ssirica0/multi-gpu-collectives"

# ===================== Validazione =====================
if [[ "$LIBRARY" != "nccl" && "$LIBRARY" != "oneccl" && "$LIBRARY" != "mpi" ]]; then
    echo "Errore: library deve essere 'nccl', 'oneccl' o 'mpi', ricevuto: '$LIBRARY'"
    exit 1
fi

BINARY="${PROJECT_ROOT}/build/${LIBRARY}/init_time"
if [ ! -x "$BINARY" ]; then
    echo "Errore: binario non trovato o non eseguibile: $BINARY"
    echo "Assicurati di aver compilato con: make ${LIBRARY}"
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

# Rimuovi CSV esistente: ogni lancio ricomincia da zero
CSV_FILE="${OUTPUT_DIR}/${LIBRARY}_init_time_${NUM_GPUS}ranks_results.csv"
if [ -f "$CSV_FILE" ]; then
    echo "Rimuovo CSV precedente: $CSV_FILE"
    rm "$CSV_FILE"
fi

# ===================== Header =====================
echo "============================================================"
echo " Cluster:    Leonardo (CINECA)"
echo " Init Time:  ${LIBRARY}"
echo " Ranks:      ${NUM_GPUS} | Iterazioni: ${NUM_ITERS}"
echo " Nodo:       $(hostname)"
echo " Job ID:     ${SLURM_JOB_ID}"
echo " Output:     ${OUTPUT_DIR}"
echo "============================================================"
echo ""

FAILED=0

for iter in $(seq 1 "$NUM_ITERS"); do
    echo -n "  Esecuzione [${iter}/${NUM_ITERS}]: "
    if ! srun --ntasks="${NUM_GPUS}" \
              "${BINARY}" --iter "$iter" --output "$OUTPUT_DIR" 2>&1; then
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
