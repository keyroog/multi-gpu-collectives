# Scatter Collective

This directory contains an implementation of the Scatter collective operation using Intel OneCCL.

## Description

Scatter is a collective communication operation where one process (the "root") distributes different segments of data to all processes (including itself). It's a one-to-many communication pattern that is the complement of Gather - instead of collecting data from all processes, it distributes unique data to each process.

## Operation Details

- **Input**: Only the root rank provides `count * size` elements, segmented for distribution
- **Output**: All ranks receive `count` elements (their assigned segment)
- **Communication Pattern**: One-to-many distribution from root rank
- **Data Distribution**: Root's data is divided into equal segments, one per rank
- **Complexity**: O(log n) for tree-based implementations

## Data Flow

```
Before Scatter (Root = Rank 0):
Rank 0: [seg0][seg1][seg2][seg3]  ← Root has all data segments
Rank 1: [undefined]               ← Empty receive buffer
Rank 2: [undefined]               ← Empty receive buffer  
Rank 3: [undefined]               ← Empty receive buffer

After Scatter:
Rank 0: [seg0]  ← Root keeps segment 0
Rank 1: [seg1]  ← Receives segment 1
Rank 2: [seg2]  ← Receives segment 2
Rank 3: [seg3]  ← Receives segment 3
```

Where each segment contains `count` elements with pattern:
- seg0: [root×1000+0×100+0, root×1000+0×100+1, ...]
- seg1: [root×1000+1×100+0, root×1000+1×100+1, ...]
- seg2: [root×1000+2×100+0, root×1000+2×100+1, ...]
- seg3: [root×1000+3×100+0, root×1000+3×100+1, ...]

## Usage

```bash
icpx -o scatter scatter.cpp -lccl -lmpi -fsycl
mpirun -n <num_processes> ./scatter --dtype <type> --count <elements> --output <dir> --root <root_rank>
```

### Parameters

- `--dtype`: Data type (int, float, double)
- `--count`: Number of elements per rank (default: 10M)
- `--output`: Output directory for logging
- `--root`: Root rank that scatters the data (default: 0, must be valid rank 0 ≤ root < size)

### Examples

```bash
# Scatter from rank 0 (default) with 1M float elements per rank
mpirun -n 4 ./scatter --dtype float --count 1000000 --output ./results

# Scatter from rank 1 with 500K int elements per rank
mpirun -n 4 ./scatter --dtype int --count 500000 --output ./results --root 1
```

## Implementation Notes

- Uses SYCL for GPU memory management and computation
- Root rank allocates `count * size` elements in send buffer
- All ranks allocate `count` elements in receive buffer
- All ranks perform correctness verification on their received data
- Measures and logs execution time on all ranks
- Data pattern: root rank `r` sends `(r×1000 + dest×100 + index)` to rank `dest`
- Each rank verifies it received the correct segment intended for it

## Memory Usage

- **Root rank**: `count * size` elements for send buffer + `count` for receive buffer
- **Non-root ranks**: `count * size` elements for send buffer (unused) + `count` for receive buffer
- **Optimization note**: Non-root ranks could avoid allocating send buffer in production code

## Use Cases

- **Data distribution**: Distributing different datasets to worker processes
- **Task assignment**: Assigning different work items to each process
- **Load balancing**: Distributing computational work across processes
- **Initialization**: Setting up different initial conditions per process
- **Domain decomposition**: Distributing different parts of a problem domain

## Performance Characteristics

- **Memory usage**: O(count × size) on root, O(count) on others for meaningful data
- **Communication volume**: O(count × size) total data transmitted
- **Communication pattern**: Tree-based distribution (typically O(log n) steps)
- **Scalability**: Efficient for large process counts due to tree topology
- **Bandwidth utilization**: More efficient than individual sends from root

## Comparison with Broadcast

| Aspect | Scatter | Broadcast |
|--------|---------|-----------|
| Data per rank | Different segments | Same data |
| Root input size | count × size | count |
| Rank output size | count | count |
| Use case | Data distribution | Parameter sharing |
| Memory efficiency | Higher total usage | Lower total usage |

## Relationship to Other Collectives

- **Gather**: Inverse operation - collects different data from all ranks to root
- **AllGather**: Scatter + AllGather = each rank gets all segments
- **AllToAll**: Generalization where each rank can scatter to all others
