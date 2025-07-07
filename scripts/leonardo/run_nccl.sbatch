#!/bin/bash
#SBATCH --job-name=nccl_${1}_${2}_${3}
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/nccl_${1}_${2}_${3}_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs/nccl_${1}_${2}_${3}_%j.err
#SBATCH --exclusive

# Imposta l’ambiente NCCL fornito dal tutor
source /leonardo/home/userexternal/ssirica0/multi-gpu-collectives/scripts/leonardo/set_nccl_env.sh
export NCCL_DEBUG=INFO

# Esecuzione
srun ./build/nccl/$1 \
     --dtype $2 \
     --count $3 \
     --output /leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs
