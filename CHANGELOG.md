# Changelog

All notable changes to the Multi-GPU Collectives benchmark suite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-30

### 🎯 Major Enhancement: Goodput Analysis & Multi-Run Tracking

This release implements comprehensive tutor feedback focusing on run-level tracking, environment logging, goodput analysis, and GPU topology investigation.

### Added

#### Enhanced Logging System
- **Global run counter**: Static tracking of complete benchmark executions (not individual measurements)
- **Environment capture**: Automatic logging of performance-relevant environment variables
  - `CCL_LOG_LEVEL`, `NCCL_DEBUG`, `OMP_NUM_THREADS`, `ZE_AFFINITY_MASK`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER`
- **Extended CSV format**: Added `run_number` and `environment` columns
  - New format: `timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment`
- **Backward compatibility**: Automatic handling of legacy CSV files without new columns

#### Goodput Analysis Implementation
- **Worst-rank timing calculation**: `calculate_goodput()` method for realistic performance assessment
- **Enhanced summary logging**: Display goodput metrics alongside traditional statistics
- **Performance rationale**: Focus on slowest rank as it determines collective operation completion time

#### GPU Topology Investigation
- **Intel GPU Max analysis**: `investigate_gpu_topology.sh` script for hardware topology investigation
- **Level-Zero device enumeration**: Tools for GPU device discovery and configuration analysis
- **Xe Link connectivity**: Scripts to analyze GPU interconnect topology
- **Hardware optimization guidance**: Environment variable recommendations for Intel GPU Max

#### Advanced Analysis Tools
- **Goodput-focused visualization**: `generate_goodput_plots.py` for worst-rank timing analysis
- **Run consistency tracking**: Multi-run performance variation analysis
- **Environment impact analysis**: How environment variables affect performance
- **Enhanced benchmark script**: `run_goodput_benchmark.sh` with multiple runs support

#### Python Analysis Enhancement
- **Comprehensive backward compatibility**: Both `analyze_results.py` and `generate_plots.py` handle legacy data
- **Virtual environment setup**: Complete Python environment with required packages
- **Advanced statistical analysis**: Goodput trends, consistency metrics, environment correlation

### Changed

#### Logger Class Architecture
- **Static run counter**: Changed from instance-based to static `global_run_counter`
- **Run ID capture**: Each Logger instance captures current run ID at construction
- **Method additions**: 
  - `start_new_run()`: Static method to initiate new benchmark runs
  - `capture_environment()`: Automatic environment variable detection
  - `calculate_goodput()`: Worst-rank timing calculation
  - `log_gpu_topology_info()`: Hardware analysis logging

#### CSV Data Structure
- **Schema extension**: Added `run_number` and `environment` columns
- **Automatic compatibility**: Legacy files auto-detected and enhanced with missing columns
- **Data integrity**: Consistent run numbering across all measurements in same execution

#### Analysis Scripts Enhancement
- **Universal compatibility**: All scripts handle both legacy and enhanced CSV formats
- **Advanced metrics**: Goodput calculation and consistency analysis
- **Environment tracking**: Impact analysis of environment variables on performance
- **Visualization improvements**: Enhanced plots with goodput focus

### Technical Implementation Details

#### Static Run Management
```cpp
// New architecture
class Logger {
private:
    static int global_run_counter;  // Shared across all instances
    int current_run_id;            // Captured at construction
    
public:
    static void start_new_run() { global_run_counter++; }
    static int get_current_run_id() { return global_run_counter; }
};
```

#### Environment Auto-Detection
```cpp
std::string capture_environment() const {
    // Automatically detect and log:
    // CCL_LOG_LEVEL, NCCL_DEBUG, OMP_NUM_THREADS, 
    // ZE_AFFINITY_MASK, ZE_ENABLE_PCI_ID_DEVICE_ORDER
}
```

#### Goodput Calculation
```cpp
static double calculate_goodput(const std::vector<double>& rank_times) {
    return *std::max_element(rank_times.begin(), rank_times.end());
}
```

### Files Modified/Added

#### Core Implementation
- `src/common/include/logger.hpp` - **Enhanced** with static run counter, environment logging, goodput calculation
- `src/oneccl/allreduce/allreduce.cpp` - **Modified** to use new Logger system with run tracking

#### Analysis Scripts
- `scripts/analyze_results.py` - **Enhanced** with environment column support and goodput metrics
- `scripts/generate_plots.py` - **Enhanced** with run consistency analysis and backward compatibility
- `scripts/generate_goodput_plots.py` - **NEW** advanced goodput-focused analysis tool

#### Infrastructure Scripts  
- `scripts/investigate_gpu_topology.sh` - **NEW** GPU topology investigation for Intel GPU Max
- `src/oneccl/allreduce/run_goodput_benchmark.sh` - **NEW** enhanced benchmark script with multiple runs

#### Python Environment
- `venv/` - **NEW** Python virtual environment with required packages
- `requirements.txt` - **Implicit** pandas, matplotlib, seaborn, numpy

#### Documentation
- `README.md` - **Completely rewritten** with comprehensive feature documentation
- `ARCHITECTURE.md` - **NEW** technical implementation documentation
- `QUICKSTART.md` - **NEW** 5-minute setup and usage guide
- `CHANGELOG.md` - **NEW** this changelog

### Testing & Validation

#### Backward Compatibility Testing
- ✅ Legacy CSV files correctly processed by all analysis scripts
- ✅ Automatic column generation for missing `run_number` and `environment`
- ✅ No breaking changes to existing workflows

#### Multi-Run Testing
- ✅ Run counter correctly increments across complete benchmark executions
- ✅ All measurements within same run get consistent `run_number`
- ✅ Environment variables correctly captured and logged

#### Analysis Validation
- ✅ Goodput calculation verified against manual worst-rank identification
- ✅ Run consistency analysis with multiple benchmark executions
- ✅ Environment impact analysis with different variable settings

### Performance Impact

#### Logging Overhead
- **Minimal impact**: Environment capture adds ~1ms per benchmark run
- **Optimized I/O**: CSV writing remains efficient with new columns
- **No measurement interference**: Timing precision maintained

#### Analysis Performance
- **Backward compatibility**: No performance regression on legacy data
- **Enhanced features**: Goodput analysis adds comprehensive metrics without slowdown
- **Memory efficient**: Large dataset handling with pandas optimization

### Migration Guide

#### For Existing Users
1. **No action required**: Existing CSV files work with all analysis scripts
2. **Enhanced features**: New runs automatically include `run_number` and `environment`
3. **Updated scripts**: All analysis tools enhanced with new capabilities

#### For New Deployments
1. **Use enhanced scripts**: `run_goodput_benchmark.sh` for comprehensive analysis
2. **Environment optimization**: Follow GPU topology investigation recommendations
3. **Advanced analysis**: Utilize `generate_goodput_plots.py` for detailed insights

### Known Issues
- **Multi-node limitations**: Current run counter synchronization is single-node optimized
- **GPU topology script**: Requires execution on target Intel GPU Max cluster for full analysis
- **Environment variable scope**: Currently tracks predefined set of variables

### Future Roadmap
- **Multi-node support**: Enhanced run correlation across distributed nodes
- **Real-time monitoring**: Prometheus/InfluxDB integration
- **Machine learning**: Performance prediction and anomaly detection
- **Advanced topology**: Automatic optimal GPU affinity detection

## [1.0.0] - 2025-01-29

### Initial Implementation
- Basic OneCCL, NCCL, and RCCL AllReduce benchmarks
- Simple CSV logging without run tracking
- Basic analysis scripts for performance visualization
- Original Logger class with instance-based counters

### Added
- Multi-library collective communication benchmarks
- CSV-based result logging
- Basic performance analysis and visualization
- MPI-based multi-rank testing
- Automated benchmark execution scripts

### Known Limitations (Addressed in 2.0.0)
- No run-level tracking across complete benchmark executions
- Limited environment variable capture
- No goodput (worst-rank timing) analysis
- No GPU topology investigation tools
- Basic performance analysis without consistency tracking
