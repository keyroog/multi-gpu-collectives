# Reduce Collective

This directory contains an implementation of the Reduce collective operation using Intel OneCCL.

## Description

Reduce is a collective communication operation where all processes contribute data, and one process (the "root") receives the combined result. It's a many-to-one communication pattern that performs a reduction operation (like sum, max, min) across all contributions.

## Operation Details

- **Input**: All ranks contribute `count` elements
- **Output**: Only the root rank receives `count` reduced elements
- **Communication Pattern**: Many-to-one reduction to root rank
- **Operation**: Sum reduction (can be extended to other operations like max, min, etc.)
- **Complexity**: O(log n) for tree-based implementations

## Data Flow

```
Before Reduce (Sum):
Rank 0: [1×1, 1×2, 1×3, ...]  ← Contributes 1×(i+1)
Rank 1: [2×1, 2×2, 2×3, ...]  ← Contributes 2×(i+1)  
Rank 2: [3×1, 3×2, 3×3, ...]  ← Contributes 3×(i+1)
Rank 3: [4×1, 4×2, 4×3, ...]  ← Contributes 4×(i+1)

After Reduce to Root (e.g., Rank 0):
Rank 0: [10×1, 10×2, 10×3, ...]  ← Sum: (1+2+3+4)×(i+1)
Rank 1: [undefined results]      ← No valid data
Rank 2: [undefined results]      ← No valid data  
Rank 3: [undefined results]      ← No valid data
```

## Usage

```bash
icpx -o reduce reduce.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./reduce --dtype <type> --count <elements> --output <dir> --root <root_rank>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Number of elements to reduce (default: 10M)
- `--output`: Output directory for logging
- `--root`: Root rank that receives the reduced result (default: 0, must be valid rank 0 ≤ root < size)

### Examples

```bash
# Reduce to rank 0 (default) with 1M float elements
mpirun -n 4 ./reduce --dtype float --count 1000000 --output ./results

# Reduce to rank 2 with 500K double elements
mpirun -n 4 ./reduce --dtype double --count 500000 --output ./results --root 2
```

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- Separate send and receive buffers for clarity
- Only root rank performs correctness verification
- Measures and logs execution time on all ranks
- Each rank `r` contributes data with pattern `(r+1) × (index+1)`
- Root rank verifies result matches expected sum: `(index+1) × Σ(r+1)` for r=0 to size-1
- Non-root ranks complete operation but don't have valid results in receive buffer

## Reduction Operations

Currently implements **SUM** reduction. The OneCCL library supports other reduction operations:
- `ccl::reduction::sum` - Element-wise sum
- `ccl::reduction::max` - Element-wise maximum  
- `ccl::reduction::min` - Element-wise minimum
- `ccl::reduction::prod` - Element-wise product

## Use Cases

- **Gradient aggregation**: Summing gradients in distributed ML training
- **Global statistics**: Computing global sums, means, etc.
- **Convergence checking**: Collecting error metrics from all processes
- **Result collection**: Gathering partial results for final computation

## Performance Characteristics

- **Memory usage**: O(count) per process for input, O(count) only on root for output
- **Communication volume**: O(count) total data reduction
- **Communication pattern**: Tree-based reduction (typically O(log n) steps)
- **Scalability**: Efficient for large process counts due to tree topology
- **Bandwidth utilization**: More efficient than individual sends to root rank

## Comparison with AllReduce

| Aspect | Reduce | AllReduce |
|--------|--------|-----------|
| Result location | Only root rank | All ranks |
| Memory usage | O(count) on root only | O(count) on all ranks |
| Use case | Centralized collection | Distributed synchronization |
| Communication volume | Same | Same |
