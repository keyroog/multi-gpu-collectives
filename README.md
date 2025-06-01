# Multi-GPU Collectives Benchmark Suite

A comprehensive benchmarking framework for testing collective communication operations across different GPU communication libraries with advanced logging, goodput analysis, and performance visualization.

## 🚀 Features

### Advanced Logging System
- **Run-level tracking**: Global run counter for tracking complete benchmark executions
- **Environment capture**: Automatic logging of performance-relevant environment variables
- **CSV output**: Structured logging with `timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment`
- **Backward compatibility**: Support for legacy CSV files without new columns

### Goodput Analysis
- **Worst-rank timing**: Calculate goodput based on the slowest rank (realistic performance)
- **Run consistency**: Track performance variations across multiple benchmark runs
- **Environment impact**: Analyze how environment variables affect performance

### GPU Topology Investigation
- **Intel GPU Max analysis**: Tools for investigating Intel GPU Max architecture
- **Xe Link connectivity**: Scripts to analyze GPU interconnect topology
- **Level-Zero enumeration**: Device discovery and configuration analysis

### Performance Visualization
- **Automatic plotting**: Generate comprehensive performance graphs
- **Multi-dimensional analysis**: Performance vs message size, data type, scaling analysis
- **Run-to-run consistency**: Track and visualize performance variations
- **Goodput-focused visualization**: Advanced worst-rank timing analysis

### Multi-Library Support
- **OneCCL**: Intel OneAPI Collective Communications Library
- **NCCL**: NVIDIA Collective Communications Library
- **RCCL**: ROCm Collective Communications Library

## 📁 Project Structure

```
multi-gpu-collectives/
├── src/
│   ├── common/
│   │   └── include/
│   │       ├── logger.hpp          # Enhanced Logger with run tracking & goodput
│   │       ├── timer.hpp           # High-precision timing utilities
│   │       └── arg_parser.hpp      # Command-line argument parsing
│   ├── oneccl/
│   │   └── allreduce/
│   │       ├── allreduce.cpp       # OneCCL AllReduce benchmark
│   │       ├── run_benchmark.sh    # Basic benchmark script
│   │       ├── run_goodput_benchmark.sh  # Advanced multi-run script
│   │       └── README.md           # OneCCL-specific documentation
│   ├── nccl/
│   │   └── allreduce/
│   │       └── allreduce.cpp       # NCCL AllReduce benchmark
│   └── rccl/
│       └── allreduce/
│           └── allreduce.cpp       # RCCL AllReduce benchmark
├── scripts/
│   ├── analyze_results.py          # Quick CSV analysis with goodput
│   ├── generate_plots.py           # Standard performance visualization
│   ├── generate_goodput_plots.py   # Advanced goodput analysis
│   └── investigate_gpu_topology.sh # GPU topology investigation
├── results/
│   ├── oneccl/                     # OneCCL CSV results
│   ├── nccl/                       # NCCL CSV results
│   ├── rccl/                       # RCCL CSV results
│   └── plots/                      # Generated visualizations and reports
└── venv/                           # Python virtual environment
```

## 🔧 Setup & Installation

### Prerequisites

#### System Requirements
- Intel GPU Max (for OneCCL benchmarks)
- NVIDIA GPU (for NCCL benchmarks) 
- AMD GPU (for RCCL benchmarks)
- MPI implementation (Intel MPI, OpenMPI, etc.)

#### Intel OneAPI (for OneCCL)
```bash
# Load Intel OneAPI environment
source /opt/intel/oneapi/setvars.sh

# Verify OneCCL availability
which icpx
```

#### Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas matplotlib seaborn numpy
```

### Compilation

#### OneCCL
```bash
cd src/oneccl/allreduce
icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl
```

#### NCCL
```bash
cd src/nccl/allreduce
nvcc -o allreduce allreduce.cpp -lnccl -lmpi -lcudart
```

#### RCCL
```bash
cd src/rccl/allreduce
hipcc -o allreduce allreduce.cpp -lrccl -lmpi
```

## 🎯 Usage

### Basic Benchmark Execution

#### Single Test
```bash
# OneCCL example
mpirun -n 2 ./allreduce --dtype float --count 1048576 --output ../../../results/oneccl

# Parameters:
# --dtype: Data type (int, float, double)
# --count: Number of elements in message
# --output: Directory for CSV results (optional)
```

#### Automated Benchmarks
```bash
# Basic multi-test benchmark
./run_benchmark.sh

# Advanced multi-run goodput benchmark (OneCCL)
./run_goodput_benchmark.sh
```

### Advanced Multi-Run Analysis

#### Environment Configuration
```bash
# Set performance debugging variables
export CCL_LOG_LEVEL=trace        # OneCCL detailed logging
export NCCL_DEBUG=TRACE          # NCCL detailed logging
export OMP_NUM_THREADS=1         # OpenMP thread control
export ZE_AFFINITY_MASK=0        # Intel GPU affinity

# Run benchmark with environment tracking
mpirun -n 4 ./allreduce --dtype float --count 4194304 --output ../../../results/oneccl
```

#### Multiple Runs for Consistency Analysis
```bash
# The enhanced logging system automatically tracks:
# - Global run counter across complete benchmark executions
# - Environment variables for each run
# - Run-to-run performance variations
```

### Performance Analysis

#### Quick Analysis
```bash
# Basic CSV analysis with goodput metrics
python3 scripts/analyze_results.py --input results/oneccl
```

#### Standard Visualization
```bash
# Generate standard performance plots
python3 scripts/generate_plots.py \
    --input results/oneccl \
    --output results/plots
```

#### Advanced Goodput Analysis
```bash
# Advanced worst-rank timing analysis
python3 scripts/generate_goodput_plots.py \
    --input results/oneccl \
    --output results/plots
```

#### GPU Topology Investigation
```bash
# Investigate Intel GPU Max topology (run on target cluster)
./scripts/investigate_gpu_topology.sh
```

## 📊 Understanding the Outputs

### CSV Data Format
```csv
timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment
20250530_142051,oneccl,allreduce,float,4194304,1048576,4,0,1,12.345,"CCL_LOG_LEVEL=trace,OMP_NUM_THREADS=1"
```

### Key Columns
- **run_number**: Global execution counter (increments for each complete benchmark run)
- **environment**: Performance-relevant environment variables
- **time_ms**: Execution time in milliseconds
- **rank**: MPI rank (goodput uses worst/max rank time)

### Goodput Concept
**Goodput** = Performance of the slowest rank (worst-case timing)
- More realistic than average performance
- Critical for understanding actual throughput
- Key metric for collective operation analysis

### Generated Reports
- **Summary Report**: Statistical overview with goodput metrics
- **Performance Plots**: Standard timing vs message size/data type
- **Goodput Analysis**: Advanced worst-rank timing visualization
- **Run Consistency**: Performance variation across multiple runs
- **Environment Impact**: How environment variables affect performance

## 🔍 Advanced Features

### Run Tracking Architecture
```cpp
// In benchmark code - start new complete run
Logger::start_new_run();  // Only call once per complete benchmark execution

// Individual measurements maintain run correlation
logger.log_result(data_type, count, num_ranks, rank, elapsed_ms);
```

### Environment Auto-Detection
The system automatically captures:
- `CCL_LOG_LEVEL` / `NCCL_DEBUG`: Communication library debug levels
- `OMP_NUM_THREADS`: OpenMP threading configuration
- `ZE_AFFINITY_MASK`: Intel GPU device affinity
- MPI process configuration

### Multi-Node Considerations
For multi-node deployments:
- Run counter synchronization across nodes
- Environment variable consistency checking
- Network topology impact analysis

## 🛠️ Troubleshooting

### Common Issues

#### CSV Column Compatibility
Legacy CSV files are automatically detected and handled:
- Missing `run_number`: Auto-generated sequential numbers
- Missing `environment`: Filled with "unknown"

#### Environment Variable Tracking
```bash
# Verify environment capture
mpirun -n 2 env | grep -E "(CCL_|NCCL_|OMP_|ZE_)"

# Check logged environment in CSV
cat results/oneccl/oneccl_allreduce_float_results.csv | cut -d',' -f11
```

#### Goodput Analysis Issues
- Ensure multiple ranks for meaningful goodput calculation
- Check for outlier ranks that may skew analysis
- Verify consistent test conditions across runs

### Performance Optimization

#### Intel GPU Max Specific
```bash
# Check GPU topology
./scripts/investigate_gpu_topology.sh

# Optimize for Xe Link connectivity
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="0.0,0.1"  # Use specific GPU tiles
```

#### Multi-Run Consistency
- Use consistent environment variables across runs
- Allow GPU warmup with initial test runs
- Check for thermal throttling in longer benchmarks

## 🎓 Tutor Feedback Implementation

This enhanced system addresses key tutor feedback:

1. ✅ **Run Number Tracking**: Global run counter tracks complete benchmark executions (not individual measurements)
2. ✅ **Environment Logging**: Automatic capture of performance-relevant environment variables
3. ✅ **Goodput Analysis**: Worst-rank timing calculation for realistic performance assessment
4. ✅ **GPU Topology**: Investigation tools for Intel GPU Max architecture analysis
5. ✅ **Advanced Visualization**: Goodput-focused analysis with run consistency tracking
6. ✅ **Backward Compatibility**: Seamless handling of legacy CSV data

## 🚧 Future Enhancements

### Multi-Node Support
- Enhanced run correlation across distributed nodes
- Network topology impact analysis
- Cross-node environment synchronization

### Advanced Analytics
- Automated performance regression detection
- Machine learning-based anomaly identification
- Predictive performance modeling

### Integration Capabilities  
- Prometheus/InfluxDB monitoring integration
- CI/CD pipeline integration for performance regression testing
- Real-time performance dashboard

## 📝 Contributing

When adding new features:
1. Maintain backward compatibility with existing CSV format
2. Update both analysis scripts for new data columns
3. Add comprehensive documentation
4. Test on multiple hardware configurations

## 📄 License

[Specify license - typically MIT or Apache 2.0 for open source projects]