# ReduceScatter Collective Operation with OneCCL and SYCL

This implementation demonstrates the ReduceScatter collective communication pattern using Intel OneCCL with SYCL for multi-GPU environments.

## Overview

ReduceScatter combines the functionality of AllReduce and Scatter operations. Each rank contributes a complete array, all contributions are element-wise reduced (summed), and then different segments of the reduced result are distributed to different ranks.

## Data Flow Pattern

ReduceScatter is the inverse of AllGather:
1. **Input**: Each rank contributes `count * size` elements
2. **Reduction**: Element-wise sum across all rank contributions
3. **Distribution**: Each rank receives `count` elements (different segments)

```
Rank 0: contributes [a0, a1, ..., aN] → receives reduced elements [0 : count-1]
Rank 1: contributes [b0, b1, ..., bN] → receives reduced elements [count : 2*count-1]
...
Rank i: contributes [i0, i1, ..., iN] → receives reduced elements [i*count : (i+1)*count-1]
```

## Operation Example (3 ranks, count=2)

```
Input contributions:
Rank 0: [10, 20, 30, 40, 50, 60]
Rank 1: [1,  2,  3,  4,  5,  6]
Rank 2: [100,200,300,400,500,600]

Element-wise sum: [111, 222, 333, 444, 555, 666]

Output distribution:
Rank 0 receives: [111, 222]    (elements 0-1)
Rank 1 receives: [333, 444]    (elements 2-3)  
Rank 2 receives: [555, 666]    (elements 4-5)
```

## Data Verification

The implementation uses a mathematical verification pattern:
- **Input data**: Rank `r` contributes `(r+1) * (element_index+1)` for each element
- **Expected result**: For element at global position `g`, the sum is `Σ(r+1)*(g+1)` for r=0 to size-1
- **Verification**: Each rank verifies its received segment matches expected sums

## Memory Requirements

ReduceScatter has moderate memory requirements:
- **Send buffer**: `count * size` elements per rank
- **Receive buffer**: `count` elements per rank  
- **Total per rank**: `count * (size + 1)` elements
- **System total**: `count * size * (size + 1)` elements

For 4 ranks with count=1M:
- Send buffer per rank: 4MB (for int32)
- Receive buffer per rank: 1MB
- Total system memory: 20MB

## Performance Characteristics

- **Communication volume**: Each rank sends `count * size`, receives `count`
- **Reduction complexity**: O(count * size) operations per rank
- **Network efficiency**: Excellent - no redundant transfers
- **Scalability**: Good - linear increase in data with rank count
- **Load balancing**: Perfect - equal computation and memory per rank

## Implementation Features

1. **SYCL GPU Computing**: All buffers and computations on GPU
2. **Correctness Verification**: Mathematical validation of reduced results
3. **Performance Measurement**: Detailed timing and throughput logging
4. **Segment Analysis**: Clear documentation of data distribution
5. **Template Support**: Works with int, float, double data types

## Usage

```bash
# Basic usage
mpirun -n 4 ./reduce_scatter --dtype float --count 100000

# Larger test with logging
mpirun -n 8 ./reduce_scatter --dtype double --count 1000000 --output ./results
```

## Command Line Arguments

- `--dtype`: Data type (int, float, double) - default: depends on build
- `--count`: Elements per rank in final result - default: 1M
- `--output`: Output directory for performance logs - default: current directory

## Algorithm Complexity

- **Time**: O(count * size) for reduction + network_latency
- **Space**: O(count * size) per rank
- **Messages**: O(log(size)) in tree-based implementations
- **Bandwidth**: O(count * size) total across network

## Use Cases

ReduceScatter is ideal for:

1. **Distributed Computing**: Dividing workload after global reduction
2. **Machine Learning**: Gradient aggregation + parameter distribution
3. **Scientific Computing**: Parallel matrix operations with row/column distribution  
4. **Signal Processing**: FFT computations with frequency domain partitioning
5. **Load Balancing**: Redistributing work based on computed statistics

## Relationship to Other Collectives

- **AllReduce**: ReduceScatter + AllGather (but more memory efficient)
- **Reduce**: ReduceScatter where only one rank gets all data
- **Scatter**: ReduceScatter without the reduction step
- **AllGather**: Inverse operation of ReduceScatter

## Performance Tips

1. **Optimal count size**: Use powers of 2 for better memory alignment
2. **Data locality**: Consider NUMA placement for large arrays
3. **Network topology**: Performance depends on interconnect bandwidth
4. **Memory bandwidth**: GPU memory bandwidth often the bottleneck
5. **Overlap opportunities**: Can overlap computation with communication in advanced implementations

ReduceScatter provides an excellent balance of functionality and efficiency, making it a cornerstone operation in many distributed computing applications.
