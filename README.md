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

### oneCCL toolchain (experimental)
- Intel oneAPI DPC++/C++ compiler (`icpx`)
- Intel oneCCL runtime and headers
- MPI implementation compatible with oneCCL
- Level Zero drivers for GPU access; toggle `--gpu_mode tile` to target sub-device tiles

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

> If the linker cannot find CUDA/NCCL/MPI libraries, ensure the relevant modules are loaded or extend the `NCCL_LIBS`/`ONECCL_LIBS` variables with `-L`/`-I` flags to your installation paths.

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
- `--count`: element count per rank (bytes = `count × sizeof(dtype)`)
- `--output`: directory that will receive `{library}_{collective}_{dtype}_{count}_results.csv`
- `--gpu_mode`: oneCCL only; `gpu` (default) or `tile` to map ranks to Xe GPU tiles

 Each run performs a warm-up + 5 timed iterations and executes a correctness check that inspects the device buffer on the host.

### oneCCL specifics

oneCCL executables live under `build/oneccl/` and mirror the NCCL CLI. A typical invocation on Level Zero hardware looks like:

```sh
source /opt/intel/oneapi/setvars.sh    # ensure DPC++/oneCCL toolchain is on PATH
mpirun -n 4 \
  build/oneccl/allreduce \
  --dtype float \
  --count 4194304 \
  --gpu_mode tile \
  --output logs/oneccl/allreduce
```

- MPI ranks are mapped to devices through Level Zero; use `--gpu_mode gpu` (default) to bind ranks to full GPUs or `--gpu_mode tile` to assign each rank to a Xe tile/sub-device.
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

## Output artefacts
### CSV logs
Each iteration appends a record to `${output_dir}/nccl_<collective>_<dtype>_<count>_results.csv` (or the oneCCL equivalent). Columns correspond to the `Logger` header:

```
timestamp, library, collective, data_type, message_size_bytes, message_size_elements,
num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, gpu_mode, time_ms
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