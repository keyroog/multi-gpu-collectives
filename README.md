# Multi-GPU Collective Communications with Intel OneCCL and SYCL

A comprehensive implementation suite of collective communication operations optimized for multi-GPU environments using Intel OneCCL (One Collective Communications Library) and SYCL.

## Overview

This project provides high-performance implementations of fundamental collective communication patterns used in distributed computing, machine learning, and scientific computing applications. All operations are implemented with GPU acceleration using SYCL and include comprehensive performance measurement, correctness verification, and detailed documentation.

## Implemented Collective Operations

### Core Collective Operations

| Operation | Description | Use Cases | Memory Pattern | Performance |
|-----------|-------------|-----------|----------------|-------------|
| **[AllReduce](src/oneccl/allreduce/)** | Global reduction with result on all ranks | ML gradient aggregation, global statistics | O(n) per rank | Bandwidth-bound |
| **[AllGather](src/oneccl/allgather/)** | Gather data from all ranks to all ranks | Feature collection, state synchronization | O(n×p) per rank | Memory-bound |
| **[AllToAll](src/oneccl/alltoall/)** | Complete data exchange between all rank pairs | Matrix transpose, data redistribution | O(n×p) per rank | Network-bound |
| **[Broadcast](src/oneccl/broadcast/)** | One-to-many data distribution | Parameter distribution, configuration sync | O(n) per rank | Latency-bound |
| **[Reduce](src/oneccl/reduce/)** | Global reduction with result on root rank | Final result aggregation, checksum | O(n) on root | Computation-bound |
| **[Scatter](src/oneccl/scatter/)** | Distribute different data to each rank | Work distribution, data partitioning | O(n) per rank | Bandwidth-bound |

### Advanced Variable-Size Operations

| Operation | Description | Complexity | Best Use Cases |
|-----------|-------------|------------|----------------|
| **[AllGatherV](src/oneccl/allgatherv/)** | Variable-size gather to all ranks | O(Σnᵢ) per rank | Dynamic load balancing, sparse data |
| **[AllToAllV](src/oneccl/alltoallv/)** | Variable-size complete exchange | O(Σnᵢⱼ) per rank | Graph algorithms, irregular patterns |
| **[ReduceScatter](src/oneccl/reduce_scatter/)** | Reduction + distribution | O(n×p) per rank | Distributed computing, parallel FFT |

*Where n = elements per rank, p = number of ranks, nᵢ = variable elements from rank i*

## Key Features

### 🚀 Performance Optimized
- **GPU Acceleration**: All computations performed on GPU using SYCL
- **Memory Management**: Efficient device memory allocation and transfers
- **Asynchronous Operations**: Non-blocking collective operations with proper dependency tracking
- **Performance Measurement**: Microsecond-precision timing with comprehensive logging

### ✅ Correctness Verification
- **Mathematical Validation**: Each operation includes algorithm-specific correctness checks
- **Unique Data Patterns**: Sophisticated verification schemes to detect data corruption
- **Error Detection**: Comprehensive error reporting for debugging and validation
- **Multiple Data Types**: Support for int, float, and double with type-specific verification

### 📊 Comprehensive Logging
- **CSV Output**: Machine-readable performance data for analysis
- **Multiple Metrics**: Execution time, throughput, message size, rank count
- **Automated Analysis**: Integration with plotting and analysis scripts
- **Scalability Studies**: Built-in support for performance scaling analysis

### 🏗️ Production Ready
- **Template-Based Design**: Generic implementation supporting multiple data types
- **Error Handling**: Robust error checking and graceful failure handling
- **Documentation**: Detailed README for each operation with usage examples
- **Extensible Architecture**: Clean separation of concerns for easy modification

## Quick Start

### Prerequisites

```bash
# Intel OneAPI Toolkit (required)
source /opt/intel/oneapi/setvars.sh

# Dependencies
# - Intel OneCCL (included in OneAPI)
# - Intel SYCL compiler (icpx)
# - MPI implementation (Intel MPI recommended)
# - Level-Zero GPU runtime
```

### Build and Run

```bash
# Clone the repository
git clone <repository-url>
cd multi-gpu-collectives

# Build any collective operation (example: AllReduce)
cd src/oneccl/allreduce
icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl

# Run with 4 GPUs
mpirun -n 4 ./allreduce --dtype float --count 1000000 --output ./results

# View results
cat ./results/oneccl_allreduce_float_1000000_results.csv
```

### Running All Operations

```bash
# Build all operations
for op in allreduce allgather alltoall broadcast reduce scatter allgatherv alltoallv reduce_scatter; do
    cd src/oneccl/$op
    icpx -o $op ${op//_/}.cpp -lccl -lmpi -fsycl
    cd ../../..
done

# Run comprehensive benchmark
./scripts/run_all_benchmarks.sh
```

## Operation Details

### Basic Collective Operations

#### AllReduce
- **Purpose**: Global reduction with result available on all ranks
- **Pattern**: All-to-all reduction
- **Use Case**: ML training (gradient aggregation)
- **Memory**: O(n) per rank
- **Performance**: Typically bandwidth-limited

#### AllGather  
- **Purpose**: Collect data from all ranks to all ranks
- **Pattern**: One-to-all, replicated on all
- **Use Case**: Feature collection, model synchronization
- **Memory**: O(n×p) per rank
- **Performance**: Memory bandwidth and network limited

#### Broadcast
- **Purpose**: One rank distributes data to all others
- **Pattern**: One-to-all
- **Use Case**: Parameter distribution, configuration updates
- **Memory**: O(n) per rank
- **Performance**: Network latency dominated

### Advanced Operations

#### AllToAllV
- **Purpose**: Complete variable-size data exchange
- **Pattern**: N×N communication matrix with variable sizes
- **Use Case**: Irregular data redistribution, graph algorithms
- **Memory**: O(Σnᵢⱼ) highly variable
- **Performance**: Network congestion prone

#### ReduceScatter
- **Purpose**: Reduction followed by distribution of segments
- **Pattern**: All-to-all reduction + segmented distribution
- **Use Case**: Distributed matrix operations, parallel FFT
- **Memory**: O(n×p) send, O(n) receive
- **Performance**: Good balance of computation and communication

## Performance Characteristics

### Scalability Analysis

| Operation | Time Complexity | Space Complexity | Network Messages | Scalability |
|-----------|----------------|------------------|------------------|-------------|
| AllReduce | O(n + log p) | O(n) | O(log p) | Excellent |
| AllGather | O(n×p) | O(n×p) | O(p) | Good |
| AllToAll | O(n×p) | O(n×p) | O(p²) | Poor |
| Broadcast | O(n + log p) | O(n) | O(log p) | Excellent |
| AllToAllV | O(Σnᵢⱼ) | O(Σnᵢⱼ) | O(p²) | Very Poor |

### Memory Requirements (4 ranks, 1M elements)

| Operation | Per-Rank Memory | Total System | Memory Efficiency |
|-----------|----------------|--------------|-------------------|
| AllReduce | 8MB | 32MB | Excellent |
| AllGather | 32MB | 128MB | Good |
| AllToAll | 32MB | 128MB | Good |
| Broadcast | 8MB | 32MB | Excellent |
| AllToAllV | 40-120MB | 320MB | Poor |

## Usage Patterns

### Command Line Interface

All operations support a consistent command-line interface:

```bash
./<operation> [options]

Options:
  --dtype TYPE     Data type: int, float, double (default: float)
  --count SIZE     Number of elements (operation-specific defaults)
  --output DIR     Directory for CSV results (default: current dir)
  --help          Show usage information
```

### Integration Examples

#### Machine Learning Training
```bash
# Gradient aggregation
mpirun -n 8 ./allreduce --dtype float --count 10000000

# Parameter broadcast
mpirun -n 8 ./broadcast --dtype float --count 10000000 --root 0
```

#### Scientific Computing
```bash
# Matrix redistribution
mpirun -n 4 ./alltoall --dtype double --count 1000000

# Distributed FFT
mpirun -n 8 ./reduce_scatter --dtype double --count 2097152
```

#### Data Processing
```bash
# Feature collection
mpirun -n 16 ./allgather --dtype float --count 500000

# Dynamic load balancing
mpirun -n 8 ./allgatherv --dtype int --count 100000
```

## Performance Analysis

### Benchmark Scripts

```bash
# Automated benchmarking
./scripts/run_comprehensive_benchmark.sh

# Generate performance plots
python3 scripts/generate_plots.py --input results/ --output plots/

# Scaling analysis
./scripts/scaling_analysis.sh --max-ranks 16 --operation allreduce
```

### Performance Metrics

The logging system captures:
- **Execution Time**: Microsecond precision timing
- **Throughput**: Elements/second and bandwidth
- **Scalability**: Performance vs. rank count
- **Memory Usage**: Device memory utilization
- **Network Efficiency**: Communication pattern analysis

## Architecture

### Directory Structure

```
multi-gpu-collectives/
├── src/
│   ├── common/include/          # Shared headers (arg_parser, logger)
│   └── oneccl/                  # OneCCL implementations
│       ├── common/              # Shared OneCCL context
│       ├── allreduce/           # AllReduce implementation
│       ├── allgather/           # AllGather implementation
│       ├── alltoall/            # AllToAll implementation
│       ├── broadcast/           # Broadcast implementation
│       ├── reduce/              # Reduce implementation
│       ├── scatter/             # Scatter implementation
│       ├── allgatherv/          # AllGatherV implementation
│       ├── alltoallv/           # AllToAllV implementation
│       └── reduce_scatter/      # ReduceScatter implementation
├── scripts/                     # Automation and analysis scripts
├── results/                     # Performance data output
└── docs/                        # Additional documentation
```

### Common Components

#### OneCCL Context (`oneccl_context.hpp`)
- MPI and CCL initialization
- GPU device selection and SYCL queue creation
- Communicator and stream setup
- Logger initialization

#### Argument Parser (`arg_parser.hpp`)
- Consistent command-line interface
- Type-safe parameter parsing
- Default value handling

#### Performance Logger (`logger.hpp`)
- CSV output formatting
- Automatic directory creation
- Thread-safe logging

## Best Practices

### Performance Optimization

1. **Memory Management**
   - Use device memory for all computations
   - Minimize host-device transfers
   - Align buffer sizes to GPU memory hierarchy

2. **Communication Patterns**
   - Choose appropriate collective for your data pattern
   - Consider memory vs. network trade-offs
   - Use variable-size operations only when necessary

3. **Scaling Considerations**
   - Test with target rank counts
   - Monitor memory usage with large messages
   - Consider network topology for large-scale deployments

### Development Guidelines

1. **Error Handling**
   - Always verify correctness with test patterns
   - Use meaningful error messages
   - Handle edge cases (empty messages, single rank)

2. **Documentation**
   - Document algorithm complexity
   - Provide usage examples
   - Include performance characteristics

3. **Testing**
   - Test with multiple data types
   - Verify with different rank counts
   - Check edge cases and error conditions

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```bash
   # Ensure OneAPI environment is loaded
   source /opt/intel/oneapi/setvars.sh
   
   # Check compiler version
   icpx --version
   ```

2. **Runtime Errors**
   ```bash
   # Check GPU availability
   sycl-ls
   
   # Verify MPI setup
   mpirun -n 2 hostname
   ```

3. **Performance Issues**
   - Monitor GPU memory usage with `nvidia-smi` or equivalent
   - Check network bandwidth with `iperf` for multi-node setups
   - Reduce message sizes for memory-constrained systems

### Debugging

Enable verbose output for debugging:
```bash
export CCL_LOG_LEVEL=debug
export SYCL_PI_TRACE=1
mpirun -n 2 ./allreduce --dtype float --count 1000
```

## Contributing

This project is designed for extensibility:

1. **Adding New Collectives**: Follow the template pattern in existing operations
2. **Performance Improvements**: Optimize SYCL kernels and memory patterns
3. **Analysis Tools**: Extend logging and plotting capabilities
4. **Documentation**: Improve operation-specific documentation

## Future Enhancements

- **Advanced Algorithms**: Tree-based and pipeline implementations
- **Multi-GPU Support**: Single-node multi-GPU optimizations
- **Network Optimization**: InfiniBand and specialized interconnect support
- **Profiling Integration**: Intel VTune and other profiling tool integration
- **Containerization**: Docker images with complete environment setup

## References

- [Intel OneCCL Documentation](https://spec.oneapi.com/oneccl/latest/)
- [SYCL Specification](https://www.khronos.org/sycl/)
- [Collective Communication Algorithms](https://htor.inf.ethz.ch/publications/img/hoefler-et-al-scientific-programming.pdf)
- [MPI Collective Operations](https://www.mcs.anl.gov/research/projects/mpi/standard.html)

---

This implementation provides a solid foundation for high-performance collective communications in GPU-accelerated distributed computing environments. Each operation is carefully optimized for performance while maintaining correctness and providing comprehensive analysis capabilities.