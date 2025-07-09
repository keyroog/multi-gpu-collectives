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

# Dimensioni fisse dei messaggi: (bytes, label)
MESSAGE_SIZES = [
    (64,           '64b'),
    (4 * 1024,     '4Kib'),
    (256 * 1024,   '256Kib'),
    (16 * 1024**2, '16Mib'),
    (256 * 1024**2,'256Mib'),
    (1 * 1024**3,  '1Gib'),
]

def main():
    parser = argparse.ArgumentParser(
        description="Sottomette job SBATCH per test NCCL con message size fisse."
    )
    parser.add_argument(
        'collective',
        choices=COLLECTIVES + ['all'],
        help="Nome della collettiva o 'all' per tutte"
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
    to_run = COLLECTIVES if args.collective == 'all' else [args.collective]

    # Crea cartella logs se non esiste
    logs_dir = '/leonardo/home/userexternal/ssirica0/multi-gpu-collectives/logs'
    os.makedirs(logs_dir, exist_ok=True)

    elem_size = DTYPES[args.dtype]
    for col in to_run:
        for msg_bytes, label in MESSAGE_SIZES:
            cnt = msg_bytes // elem_size
            # Job name con etichetta umana esatta
            job_name = f"nccl_{col}_{args.dtype}_{label}"
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