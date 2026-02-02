# Multi-GPU Collectives Benchmarks

A benchmarking suite for multi-GPU collective communication built around NVIDIA NCCL (CUDA) and Intel oneCCL (SYCL). The project targets HPC clusters and bundles build scripts, batch submission helpers, and plotting utilities to take results from compilation to visual analysis.

## Highlights
- Eight classic collective patterns (`allgather`, `allreduce`, `alltoall`, `broadcast`, `gather`, `reduce`, `reduce_scatter`, `scatter`) implemented for NCCL and scaffolded for oneCCL
- Deterministic CLI interface shared by every binary (`--dtype`, `--count`, optional `--output`)
- SLURM workflow for sweeping message sizes across collectives with reproducible log files
- Python tooling to aggregate CSV logs and generate per-collective performance plots
- Rich CSV logging (per rank, per iteration) including node topology hints for multi-node runs
- oneCCL back-end supports both full-device and tile-level placement via `--gpu_mode`

## Repository Layout
```
Makefile                     # Builds NCCL and oneCCL binaries (one executable per collective)
requirements.txt             # Python dependencies (plotting)
logs/                        # Sample CSV outputs and SLURM logs (git-ignored in practice)
plots/                       # Example rendered plots
scripts/
  leonardo/                  # SLURM template & environment setup for Leonardo cluster
    run_nccl.sbatch
    set_nccl_env.sh
  unisa-hpc/                 # Benchmark scripts for UNISA-HPC cluster
    run_benchmark.sh         # Message-size sweep runner (interactive sessions)
  plots/
    plot_nccl_results.py     # CSV aggregation & visualization
  python/
    nccl_exec.py             # Batch submitter for fixed message sizes
src/
  common/include/            # Arg parser & structured logger shared across back-ends
  nccl/                      # CUDA implementations (one folder per collective)
    common/nccl_context.hpp  # MPI/NCCL bootstrap and logging wiring
  oneccl/                    # SYCL/oneCCL implementations (work in progress)
    common/oneccl_context.hpp
```

## Prerequisites

### NCCL toolchain
- NVIDIA GPU with recent drivers
- CUDA toolkit providing `nvcc`
- NCCL runtime & development headers
- MPI implementation (OpenMPI, MPICH, etc.)
- NVIDIA Management Library (`libnvidia-ml.so`), automatically linked via the Makefile

### oneCCL toolchain
- DPC++/C++ compiler with SYCL support (`icpx` or custom `clang++` with `-fsycl`)
- Intel oneCCL runtime and headers
- MPI implementation compatible with oneCCL
- GPU drivers (Level Zero or CUDA via SYCL); toggle `--gpu_mode tile` to target sub-device tiles where supported

### Python utilities
- Python ≥ 3.8
- `pip install -r requirements.txt` (installs `pandas`, `matplotlib`, `seaborn`)

## Building the benchmarks
The Makefile emits one binary per collective under `build/nccl/` (or `build/oneccl/`). It auto-creates build directories on demand.

```sh
# build every NCCL collective
make nccl

# build a single NCCL target
make nccl nccl_collective=allreduce

# build every oneCCL collective (sources exist but some implementations are incomplete)
make oneccl

# clean all artifacts
make clean
```

The Makefile picks up toolchain paths from environment variables. Set them before building:

| Variable | Purpose | Default |
|---|---|---|
| `NVCC` | CUDA compiler | `nvcc` |
| `NCCL_ROOT` | NCCL install prefix (expects `include/`, `lib/`) | _(none)_ |
| `DPCPP_CLANGXX` | DPC++ compiler with SYCL support | falls back to `icpx` |
| `DPCPP_LIB` | DPC++ runtime library path | _(none)_ |
| `ONECCL_INSTALL` | oneCCL install prefix | _(none)_ |
| `SYCL_TARGET` | `-fsycl-targets` value | `nvptx64-nvidia-cuda` |

## Running benchmarks manually
Every binary expects at least the data type and element count. Optionally pass `--output` to persist CSV results.

```sh
# 4 ranks, 4 GPUs (example using SLURM srun)
srun -N 1 -n 4 --gpus-per-node=4 \
  build/nccl/allreduce \
  --dtype float \
  --count 4194304 \
  --output logs/nccl/allreduce
```

CLI switches:
- `--dtype`: `int`, `float`, or `double`
- `--count`: **global** element count across all ranks (bytes = `count × sizeof(dtype)`). The program divides internally by `num_ranks` for allreduce or by `num_ranks²` for alltoall
- `--output`: directory that will receive `{library}_{collective}_{dtype}_{count}_results.csv`
- `--gpu_mode`: oneCCL only; `gpu` (default) or `tile` to map ranks to Xe GPU tiles

Each invocation performs one warm-up call (not timed) followed by a single timed iteration, plus a correctness check that copies the device buffer back to the host. To collect multiple samples, invoke the binary several times (the benchmark script does this automatically).

### oneCCL specifics

oneCCL executables live under `build/oneccl/` and mirror the NCCL CLI. A typical invocation looks like:

```sh
source /opt/intel/oneapi/setvars.sh    # ensure DPC++/oneCCL toolchain is on PATH
mpirun -n 4 \
  build/oneccl/allreduce \
  --dtype float \
  --count 4194304 \
  --gpu_mode tile \
  --output logs/oneccl/allreduce
```

- MPI ranks are mapped to devices using the node-local rank; use `--gpu_mode gpu` (default) to bind ranks to full GPUs or `--gpu_mode tile` to assign each rank to a sub-device tile.
- The logging pipeline is identical to NCCL, so existing plotting scripts can ingest oneCCL CSVs alongside NCCL results.

## Batch submission workflow (SLURM)
Automate message-size sweeps with the helper script under `scripts/python/`.

```sh
# Install Python dependencies once
pip install -r requirements.txt

# Submit all collectives for float datatype using the Leonardo template
python scripts/python/nccl_exec.py all float \
  --sbatch-script scripts/leonardo/run_nccl.sbatch

# Submit a single collective in double precision
python scripts/python/nccl_exec.py allreduce double \
  --sbatch-script scripts/leonardo/run_nccl.sbatch
```

Defaults baked into `nccl_exec.py`:
- Message sizes: 64 B, 4 KiB, 256 KiB, 16 MiB, 256 MiB, 1 GiB
- Absolute paths to `/leonardo/home/userexternal/ssirica0/...` for both the sbatch template and log directory

> Update those paths (and the `#SBATCH` metadata inside `scripts/leonardo/run_nccl.sbatch`) to match your cluster. The sbatch template sources `set_nccl_env.sh` to load CUDA/MPI/NCCL modules before launching `srun`.

## Interactive benchmark runner (UNISA-HPC)

For clusters where you work inside interactive `srun` sessions, use the dedicated benchmark script:

```sh
# 1. Get an interactive session with GPUs
srun --partition gpuq -A usershpc --gres=gpu:4 -N 1 -n 4 --pty bash

# 2. Load environment
source $HOME/scripts/export_variables.sh

# 3. Run benchmarks
scripts/unisa-hpc/run_benchmark.sh nccl allreduce float
scripts/unisa-hpc/run_benchmark.sh oneccl allreduce float
scripts/unisa-hpc/run_benchmark.sh nccl alltoall float
scripts/unisa-hpc/run_benchmark.sh oneccl alltoall float
```

The script sweeps 11 message sizes (1 B to 1 GiB) and runs 5 iterations per size. Sizes that are too small for the chosen dtype/collective/GPU count are skipped automatically.

Environment variables for customisation:

| Variable | Purpose | Default |
|---|---|---|
| `NUM_GPUS` | Number of MPI ranks / GPUs | `4` |
| `NUM_ITERS` | Iterations per message size | `5` |
| `PROJECT_ROOT` | Project root path | `$HOME/multi-gpu-collectives` |

Results are written to `results/unisa-hpc/{nccl,oneccl}/`.

## Output artefacts
### CSV logs
Each iteration appends a record to `${output_dir}/nccl_<collective>_<dtype>_<count>_results.csv` (or the oneCCL equivalent). Columns correspond to the `Logger` header:

```
timestamp, library, collective, data_type, message_size_bytes, message_size_elements,
num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, gpu_mode, test_passed, time_ms
```

### SLURM stdout/stderr
`scripts/leonardo/run_nccl.sbatch` uses `%x_%j` naming, producing files such as `logs/nccl_allreduce_float_16Mib_<jobid>.out` and `.err`.

## Plotting results
Generate per-collective PNGs with the plotting utility:

```sh
python scripts/plots/plot_nccl_results.py \
  --logs-dir logs/nccl \
  --out-dir plots/nccl
```

- X-axis: message size (log₂ scale with readable tick labels)
- Y-axis: mean time (ms) in log scale, averaged over all ranks and iterations
- Hue: data type

The script filters to the canonical message sizes listed above; additional sizes will be ignored unless you extend `PLOT_SIZES`.