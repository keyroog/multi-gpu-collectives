# Quick Start Guide - Multi-GPU Collectives

## ⚡ 5-Minute Setup

### 1. Prerequisites Check
```bash
# Check Intel OneAPI (for OneCCL)
source /opt/intel/oneapi/setvars.sh
which icpx

# Check MPI
which mpirun

# Check Python
python3 --version
```

### 2. Build OneCCL Benchmark
```bash
cd src/oneccl/allreduce
icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl
```

### 3. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn numpy
```

### 4. Run First Benchmark
```bash
# Single test
mpirun -n 2 ./allreduce --dtype float --count 1048576 --output ../../../results/oneccl

# Automated multi-test
./run_benchmark.sh
```

### 5. Analyze Results
```bash
# Quick overview
python3 ../../../scripts/analyze_results.py --input ../../../results/oneccl

# Generate plots
python3 ../../../scripts/generate_plots.py \
    --input ../../../results/oneccl \
    --output ../../../results/plots
```

## 🎯 Common Use Cases

### Performance Benchmarking
```bash
# Test different message sizes and data types
./run_benchmark.sh

# Check results
ls ../../../results/oneccl/
cat ../../../results/plots/summary_report.txt
```

### Multi-Run Consistency Analysis
```bash
# Run advanced goodput benchmark (multiple runs)
./run_goodput_benchmark.sh

# Advanced goodput analysis
python3 ../../../scripts/generate_goodput_plots.py \
    --input ../../../results/oneccl \
    --output ../../../results/plots
```

### Environment Impact Testing
```bash
# Test with different environment settings
export CCL_LOG_LEVEL=trace
export OMP_NUM_THREADS=1
mpirun -n 4 ./allreduce --dtype float --count 4194304 --output ../../../results/oneccl

# Reset environment
unset CCL_LOG_LEVEL OMP_NUM_THREADS
mpirun -n 4 ./allreduce --dtype float --count 4194304 --output ../../../results/oneccl

# Compare environment impact
python3 ../../../scripts/analyze_results.py --input ../../../results/oneccl
```

### GPU Topology Investigation
```bash
# Run topology investigation (on target cluster)
../../../scripts/investigate_gpu_topology.sh
```

## 📊 Understanding Output

### CSV Data Structure
Every benchmark run generates CSV files with this format:
```csv
timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment
20250530_142051,oneccl,allreduce,float,4194304,1048576,4,0,1,12.345,"CCL_LOG_LEVEL=trace,OMP_NUM_THREADS=1"
```

### Key Metrics
- **time_ms**: Individual rank execution time
- **run_number**: Complete benchmark execution counter
- **Goodput**: Worst-rank timing (most realistic performance)
- **Environment**: Performance-relevant environment variables

### Generated Visualizations
- `*_performance_vs_message_size.png`: Timing vs message size
- `*_performance_by_datatype.png`: Data type comparison
- `*_scaling_analysis.png`: Multi-rank scaling (if available)
- `*_run_consistency.png`: Performance variation across runs
- `summary_report.txt`: Statistical overview

## 🔧 Troubleshooting

### Build Issues
```bash
# OneCCL compilation error
source /opt/intel/oneapi/setvars.sh  # Ensure OneAPI is loaded
which icpx  # Verify compiler is available

# MPI issues
which mpirun  # Verify MPI is available
mpirun --version  # Check MPI version
```

### Runtime Issues
```bash
# Directory creation issues
mkdir -p results/oneccl  # Manual directory creation

# Permission issues
chmod +x run_benchmark.sh  # Make scripts executable
```

### Analysis Issues
```bash
# Python package issues
source venv/bin/activate  # Ensure virtual environment is active
pip list | grep pandas  # Verify packages are installed

# No data found
ls results/oneccl/*.csv  # Verify CSV files exist
head results/oneccl/*.csv  # Check CSV format
```

## 📈 Performance Tips

### For Intel GPU Max
```bash
# Optimize device ordering
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

# Set GPU affinity
export ZE_AFFINITY_MASK="0.0,0.1"

# Check topology first
./scripts/investigate_gpu_topology.sh
```

### For Consistent Results
```bash
# Set consistent environment
export OMP_NUM_THREADS=1
export CCL_LOG_LEVEL=error  # Reduce logging overhead

# Allow GPU warmup
# Run a few test iterations before actual benchmarking
```

### For Large-Scale Testing
```bash
# Use goodput benchmark for multiple runs
./run_goodput_benchmark.sh

# Process large datasets efficiently
python3 scripts/generate_goodput_plots.py --input results/oneccl --output results/plots
```

## 🎓 Key Concepts

### Run Number vs Individual Measurements
- **Run Number**: Increments for each complete benchmark execution (all data types)
- **Individual Measurements**: Each rank/data_type combination per run
- **Use Case**: Track performance consistency across complete benchmark runs

### Goodput vs Average Performance
- **Goodput**: Time of slowest rank (worst-case timing)
- **Average**: Mean of all rank times
- **Why Goodput Matters**: Collective operations complete when slowest rank finishes

### Environment Impact
- Different environment variables can significantly affect performance
- System automatically tracks key variables: `CCL_LOG_LEVEL`, `NCCL_DEBUG`, `OMP_NUM_THREADS`, etc.
- Use environment analysis to identify optimal configurations

## 🚀 Next Steps

### Extend Testing
```bash
# Test different libraries
cd ../nccl/allreduce  # NVIDIA GPUs
cd ../rccl/allreduce  # AMD GPUs

# Test different message sizes
mpirun -n 4 ./allreduce --dtype float --count 16777216  # 16M elements
```

### Advanced Analysis
```bash
# Multi-library comparison (after running different libraries)
python3 scripts/generate_plots.py \
    --input results/ \
    --output results/comparison_plots
```

### Integration with CI/CD
```bash
# Add to automated testing pipeline
./run_goodput_benchmark.sh && \
python3 scripts/analyze_results.py --input results/oneccl --output results/ci_report.txt
```

## 📞 Support

### Documentation
- `README.md`: Complete project overview
- `ARCHITECTURE.md`: Technical implementation details
- `src/oneccl/allreduce/README.md`: OneCCL-specific documentation

### Common Commands Reference
```bash
# Essential commands
./run_benchmark.sh                    # Basic multi-test benchmark
./run_goodput_benchmark.sh           # Advanced multi-run benchmark
python3 scripts/analyze_results.py   # Quick analysis
python3 scripts/generate_plots.py    # Standard visualization
python3 scripts/generate_goodput_plots.py  # Advanced goodput analysis
./scripts/investigate_gpu_topology.sh      # GPU topology investigation
```

Happy benchmarking! 🚀
