#!/bin/bash
#SBATCH --job-name=nccl            # nome generico, poi sovrascritto in sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/%x_%j.err
#SBATCH --exclusive

# Imposta l’ambiente NCCL
source /leonardo/home/userexternal/ssirica0/multi-gpu-collectives/scripts/leonardo/set_nccl_env.sh
export NCCL_DEBUG=INFO

# Parametri passati in sbatch (posizionali)
COLLECTIVE=$1
DTYPE=$2
COUNT=$3

# Esecuzione
srun /leonardo/home/userexternal/ssirica0/multi-gpu-collectives/build/nccl/${COLLECTIVE} \
     --dtype ${DTYPE} \
     --count ${COUNT} \
     --output /leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs