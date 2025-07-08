#!/usr/bin/env python3
import os
import subprocess
import argparse

# Mapping dei tipi dati alla loro size in byte
DTYPES = {
    'int':    4,
    'float':  4,
    'double': 8,
}

# Lista di tutte le collettive supportate
COLLECTIVES = [
    'allgather','allreduce','alltoall',
    'broadcast','gather','reduce',
    'reduce_scatter','scatter'
]

def generate_counts(elem_size):
    """Genera potenze di 2 finché size <= 1 GiB."""
    max_bytes = 1 << 30  # 1 GiB
    cnt = 1
    while cnt * elem_size <= max_bytes:
        yield cnt
        cnt *= 4

def main():
    parser = argparse.ArgumentParser(
        description="Sottomette job SBATCH per test NCCL."
    )
    parser.add_argument(
        'collective',
        help="Nome della collettiva o 'all' per tutte",
        choices=COLLECTIVES + ['all']
    )
    parser.add_argument(
        'dtype',
        choices=DTYPES.keys(),
        help="Tipo di dato (int, float, double)"
    )
    parser.add_argument(
        '--sbatch-script',
        default='/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/scripts/leonardo/run_nccl.sbatch',
        help="Percorso allo sbatch template"
    )
    args = parser.parse_args()

    # Determina le collettive da eseguire
    if args.collective == 'all':
        to_run = COLLECTIVES
    else:
        to_run = [args.collective]

    # Crea cartella logs se non esiste
    logs_dir = '/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Genera e invia i job
    elem_size = DTYPES[args.dtype]
    for col in to_run:
        for cnt in generate_counts(elem_size):
            job_name = f"nccl_{col}_{args.dtype}_{cnt}"
            cmd = [
                'sbatch',
                '--job-name', job_name,
                args.sbatch_script,
                col, args.dtype, str(cnt)
            ]
            print("Submitting:", ' '.join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
