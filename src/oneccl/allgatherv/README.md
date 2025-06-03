# AllGatherV Collective

This directory contains an implementation of the AllGatherV (AllGather Variable) collective operation using Intel OneCCL.

## Description

AllGatherV is a collective communication operation where each process contributes a variable number of data elements, and all processes receive all contributed data from all processes. Unlike regular AllGather where each process contributes the same amount, AllGatherV allows different contribution sizes per process.

## Operation Details

- **Input**: Each rank contributes a variable number of elements (different per rank)
- **Output**: All ranks receive all data from all ranks, properly organized
- **Communication Pattern**: All-to-All with variable message sizes
- **Data Layout**: Concatenated data from all ranks, with proper displacement tracking
- **Complexity**: O(n) communication rounds, but variable data volumes

## Data Distribution Pattern

In this implementation:
- Rank `r` contributes `base_count + r × 1000` elements
- Each rank's data has pattern: `r × 10000 + element_index`

### Example with 4 ranks and base_count=2000:

```
Before AllGatherV:
Rank 0: [0, 1, 2, ..., 1999]           (2000 elements)
Rank 1: [10000, 10001, ..., 12999]     (3000 elements) 
Rank 2: [20000, 20001, ..., 23999]     (4000 elements)
Rank 3: [30000, 30001, ..., 34999]     (5000 elements)

After AllGatherV (all ranks have):
[0, 1, ..., 1999, 10000, 10001, ..., 12999, 20000, 20001, ..., 23999, 30000, 30001, ..., 34999]
 ↑                ↑                      ↑                      ↑
 Rank 0 data      Rank 1 data           Rank 2 data           Rank 3 data
 (2000 elems)     (3000 elems)          (4000 elems)          (5000 elems)
```

## Usage

```bash
icpx -o allgatherv allgatherv.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./allgatherv --dtype <type> --count <base_elements> --output <dir>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Base number of elements per rank (default: 1M, each rank adds rank×1000 more)
- `--output`: Output directory for logging

### Examples

```bash
# Run with 4 processes, base 500K elements (rank 0: 500K, rank 1: 501K, rank 2: 502K, rank 3: 503K)
mpirun -n 4 ./allgatherv --dtype float --count 500000 --output ./results

# Run with variable data sizes
mpirun -n 8 ./allgatherv --dtype int --count 100000 --output ./results
```

## Implementation Details

### Memory Management
- **Send buffer**: Each rank allocates `base_count + rank × 1000` elements
- **Receive buffer**: Each rank allocates total elements from all ranks
- **Counts array**: Tracks how many elements each rank contributes
- **Displacements array**: Tracks where each rank's data starts in receive buffer

### Algorithm Steps
1. Calculate variable send counts per rank
2. Compute receive counts and displacement offsets
3. Allocate appropriately sized buffers
4. Initialize send data with rank-specific patterns
5. Perform AllGatherV operation
6. Verify data integrity and placement

## Performance Characteristics

- **Memory usage**: O(total_elements) where total varies per rank
- **Communication volume**: Sum of all ranks' contributions
- **Load balancing**: Naturally unbalanced due to variable data sizes
- **Scalability**: Depends on data distribution pattern

## Use Cases

- **Irregular data collection**: When processes have different amounts of results
- **Load balancing scenarios**: After work redistribution
- **Sparse data gathering**: When some ranks have more relevant data
- **Dynamic content sharing**: Variable-length messages or arrays
- **Heterogeneous workloads**: Different computational outputs per rank

## Comparison with Regular AllGather

| Aspect | AllGather | AllGatherV |
|--------|-----------|------------|
| Input size per rank | Fixed (count) | Variable (counts[i]) |
| Memory predictability | High | Variable |
| Implementation complexity | Simple | Moderate |
| Use cases | Uniform data | Non-uniform data |
| Buffer management | Straightforward | Requires displacement calculation |

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- Includes comprehensive correctness verification
- Measures and logs execution time
- Provides detailed data distribution information
- Handles variable-size contributions efficiently
- All ranks verify the complete gathered data

## Advanced Features

- **Dynamic sizing**: Each rank determines its contribution at runtime
- **Displacement tracking**: Automatic calculation of data positions
- **Comprehensive verification**: Validates both data content and placement
- **Performance logging**: Reports per-rank timing and data volumes
- **Debug information**: Shows data distribution across ranks
