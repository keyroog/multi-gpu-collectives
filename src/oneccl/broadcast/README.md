# Broadcast Collective

This directory contains an implementation of the Broadcast collective operation using Intel OneCCL.

## Description

Broadcast is a collective communication operation where one process (the "root") sends the same data to all other processes. It's a one-to-many communication pattern, fundamental for distributing parameters, configuration data, or initial values across all processes.

## Operation Details

- **Input**: Only the root rank contributes `count` elements
- **Output**: All ranks (including root) have the same `count` elements
- **Communication Pattern**: One-to-many from root rank to all other ranks
- **Complexity**: O(log n) for tree-based implementations, O(n) communication volume

## Data Flow

```
Before Broadcast:
Rank 0: [data0, data1, data2, ...]  ← Root rank
Rank 1: [  -1,   -1,   -1, ...]    ← Uninitialized
Rank 2: [  -1,   -1,   -1, ...]    ← Uninitialized
Rank 3: [  -1,   -1,   -1, ...]    ← Uninitialized

After Broadcast:
Rank 0: [data0, data1, data2, ...]  ← Same data
Rank 1: [data0, data1, data2, ...]  ← Root's data
Rank 2: [data0, data1, data2, ...]  ← Root's data
Rank 3: [data0, data1, data2, ...]  ← Root's data
```

## Usage

```bash
icpx -o broadcast broadcast.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./broadcast --dtype <type> --count <elements> --output <dir> --root <root_rank>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Number of elements to broadcast (default: 10M)
- `--output`: Output directory for logging
- `--root`: Root rank that broadcasts the data (default: 0, optional, must be valid rank 0 ≤ root < size)

### Examples

```bash
# Broadcast from rank 0 (default) with 1M float elements
mpirun -n 4 ./broadcast --dtype float --count 1000000 --output ./results

# Broadcast from rank 2 with 500K int elements
mpirun -n 4 ./broadcast --dtype int --count 500000 --output ./results --root 2
```

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- In-place operation: uses same buffer for send and receive
- Includes correctness verification across all ranks
- Measures and logs execution time
- Root rank initializes data with pattern `(root_rank * 1000 + index)`
- Non-root ranks verify they received correct data from root
- Supports any valid rank as root (0 ≤ root < size)

## Use Cases

- **Parameter distribution**: Broadcasting model parameters in distributed ML
- **Configuration sharing**: Distributing configuration data to all workers
- **Initialization**: Setting initial values across all processes
- **Synchronization**: Distributing control information or flags

## Performance Characteristics

- **Memory usage**: O(count) per process
- **Communication volume**: O(count) total data transmitted
- **Communication pattern**: Tree-based (typically O(log n) steps)
- **Scalability**: Efficient for large process counts due to tree topology
- **Bandwidth utilization**: More efficient than point-to-point sends to each rank
