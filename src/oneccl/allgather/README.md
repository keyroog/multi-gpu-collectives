# AllGather Collective

This directory contains an implementation of the AllGather collective operation using Intel OneCCL.

## Description

AllGather is a collective communication operation where each process contributes data and all processes receive the contributed data from all processes. The result is that each process has a complete copy of all data from all processes.

## Operation Details

- **Input**: Each rank contributes `count` elements
- **Output**: Each rank receives `count * size` elements (where `size` is the number of processes)
- **Data Layout**: The output buffer contains data from rank 0, followed by data from rank 1, and so on

## Usage

```bash
icpx -o allgather allgather.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./allgather --dtype <type> --count <elements> --output <dir>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Number of elements per rank (default: 10M)
- `--output`: Output directory for logging

### Example

```bash
# Run with 4 processes, 1M float elements per process
mpirun -n 4 ./allgather --dtype float --count 1000000 --output ./results
```

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- Includes correctness verification
- Measures and logs execution time
- Each rank contributes unique data (rank_id * 100 + element_index) for verification
