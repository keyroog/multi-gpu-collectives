# AllToAll Collective

This directory contains an implementation of the AllToAll collective operation using Intel OneCCL.

## Description

AllToAll is a collective communication operation where each process sends different data to every other process (including itself). It's like a "transpose" operation across processes - each process contributes N segments of data, and each segment goes to a different destination process.

## Operation Details

- **Input**: Each rank contributes `count * size` elements, divided into `size` segments of `count` elements each
- **Output**: Each rank receives `count * size` elements, with segment `i` containing data from rank `i`
- **Communication Pattern**: Rank `i` sends segment `j` to rank `j`, and receives segment `i` from all ranks
- **Complexity**: O(n²) communication pattern - most intensive collective operation

## Data Layout

### Send Buffer Layout (per rank):
```
[segment_0][segment_1][segment_2]...[segment_n-1]
    ↓         ↓         ↓              ↓
  to rank0  to rank1  to rank2    to rank(n-1)
```

### Receive Buffer Layout (per rank):
```
[segment_0][segment_1][segment_2]...[segment_n-1]
    ↓         ↓         ↓              ↓
 from rank0 from rank1 from rank2  from rank(n-1)
```

## Usage

```bash
icpx -o alltoall alltoall.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./alltoall --dtype <type> --count <elements> --output <dir>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Number of elements per segment (default: 1M, smaller than other collectives due to O(n²) scaling)
- `--output`: Output directory for logging

### Example

```bash
# Run with 4 processes, 100K elements per segment
mpirun -n 4 ./alltoall --dtype float --count 100000 --output ./results
```

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- Includes correctness verification with unique data patterns
- Measures and logs execution time
- Data pattern: rank `r` sends `(r*1000 + dest*100 + index)` to destination rank `dest`
- Most memory-intensive collective: each process handles `count * size²` total data movement
- Consider using smaller `count` values for large process counts due to quadratic scaling

## Performance Considerations

- Memory usage: O(n²) where n is the number of processes
- Communication volume: Each process sends and receives `count * size` elements
- Network bandwidth: All-to-all communication pattern can saturate network
- Recommended for applications that truly need this communication pattern (matrix transpose, FFT, etc.)
