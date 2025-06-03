# AllToAllV Collective Operation with OneCCL and SYCL

This implementation demonstrates the AllToAllV (All-to-All Variable) collective communication pattern using Intel OneCCL with SYCL for multi-GPU environments.

## Overview

AllToAllV is the most flexible and memory-intensive collective operation. Each rank can send a different number of elements to each other rank, and similarly receive different amounts from each rank. This creates a complete variable-size N×N communication matrix.

## Data Flow Pattern

Unlike AllToAll where all ranks exchange equal amounts of data, AllToAllV allows each rank pair to have unique send/receive counts:

```
Rank i sends (base_count + i*100 + j*50) elements to rank j
Rank i receives (base_count + j*100 + i*50) elements from rank j
```

This creates an asymmetric communication pattern where:
- Higher-numbered ranks send progressively more data
- Data volumes vary significantly between rank pairs
- Memory requirements can be very large (O(n²) scaling)

## Communication Matrix Example (4 ranks, base_count=1000)

```
From/To    To_0    To_1    To_2    To_3
From_0:    1000    1050    1100    1150    (total sent: 4300)
From_1:    1100    1150    1200    1250    (total sent: 4700) 
From_2:    1200    1250    1300    1350    (total sent: 5100)
From_3:    1300    1350    1400    1450    (total sent: 5500)
```

Each rank receives: 4600, 4800, 5000, 5200 elements respectively.

## Data Verification

The implementation uses a sophisticated verification scheme:
- Send data encoding: `sender_rank * 100000 + dest_rank * 10000 + element_index`
- This allows verification of both source and intended destination
- Each element carries its complete routing information

## Memory Considerations

AllToAllV is extremely memory-intensive:
- **Total elements**: Sum of all send counts across all ranks
- **Memory per rank**: Can vary significantly (factor of 2-3x between ranks)
- **Scaling**: O(n²) with number of ranks
- **Buffer management**: Separate displacement arrays required

For n=8 ranks with base_count=10000:
- Rank 0: ~340KB send, ~360KB receive
- Rank 7: ~540KB send, ~560KB receive
- Total system memory: ~7.2MB

## Performance Characteristics

- **Latency**: Dominated by largest send/receive operation
- **Bandwidth**: Limited by most heavily loaded network links  
- **Scalability**: Poor due to O(n²) communication volume
- **Load balancing**: Inherently unbalanced due to variable sizes

## Implementation Features

1. **Displacement Calculation**: Automatic computation of send/recv displacement arrays
2. **Memory Management**: Efficient SYCL device buffer allocation for variable sizes
3. **Verification**: Complete correctness checking with detailed error reporting
4. **Performance Logging**: Timing and throughput measurements
5. **Communication Analysis**: Detailed matrix output showing all transfers

## Usage

```bash
# Basic usage with default parameters
mpirun -n 4 ./alltoallv --dtype float --count 5000

# Larger test with detailed output
mpirun -n 8 ./alltoallv --dtype double --count 10000 --output ./results
```

## Command Line Arguments

- `--dtype`: Data type (int, float, double) - default: depends on build
- `--count`: Base element count per rank pair - default: 10000
- `--output`: Output directory for performance logs - default: current directory

## Warning: Resource Requirements

AllToAllV requires significant memory and network resources:
- Use smaller `--count` values for testing (1000-10000)
- Monitor memory usage with large rank counts
- Network congestion likely with >16 ranks
- Consider using AllToAll for uniform communication patterns

## Algorithm Complexity

- **Time**: O(max(send_count_ij)) + network_latency  
- **Space**: O(Σ send_counts + Σ recv_counts) per rank
- **Messages**: O(n²) total transfers across all ranks
- **Network**: O(max_bandwidth_utilization) across all links

## Performance Tips

1. **Minimize base_count**: Start with small values (1000-5000)
2. **Balance communication**: Avoid extremely skewed patterns
3. **Memory locality**: Consider data placement for NUMA systems
4. **Network topology**: AllToAllV stresses interconnect heavily
5. **Use alternatives**: Consider AllGatherV + local processing when possible

## Related Operations

- **AllToAll**: Uniform version (equal counts between all rank pairs)
- **AllGatherV**: Variable gather (one-to-many with variable sizes)
- **Scatter**: One-to-many distribution
- **AllReduce**: Reduction across all ranks

AllToAllV is typically used when implementing distributed algorithms that require complete data redistribution with non-uniform workloads per rank pair.
