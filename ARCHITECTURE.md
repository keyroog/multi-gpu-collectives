# Multi-GPU Collectives - Technical Architecture

## 📋 Overview

This document describes the technical architecture of the Multi-GPU Collectives benchmark suite, focusing on the enhanced logging system, goodput analysis implementation, and multi-run tracking capabilities.

## 🏗️ Core Architecture

### Logger Class Design

#### Static Run Tracking
```cpp
class Logger {
private:
    static int global_run_counter;  // Shared across all Logger instances
    int current_run_id;             // Captured at Logger construction
    
public:
    static void start_new_run() {
        global_run_counter++;
    }
    
    static int get_current_run_id() {
        return global_run_counter;
    }
};
```

**Key Design Decisions:**
- **Static counter**: Ensures consistent run numbering across all ranks and data types
- **Captured run ID**: Each Logger instance captures the current run ID at construction
- **Thread safety**: Single-threaded MPI rank execution ensures no race conditions

#### Run Lifecycle Management

```
Benchmark Execution Flow:
┌─────────────────────────────────────────────────────────────┐
│ 1. Logger::start_new_run() (called once per complete run)  │
│ 2. Create Logger instances for each data type/test         │
│ 3. Execute tests with individual log_result() calls       │
│ 4. All measurements get same run_number                    │
│ 5. Complete run (all data types tested)                   │
└─────────────────────────────────────────────────────────────┘
```

### Environment Capture System

#### Automatic Detection
```cpp
std::string capture_environment() const {
    std::vector<std::string> env_vars = {
        "CCL_LOG_LEVEL", "NCCL_DEBUG", "OMP_NUM_THREADS", 
        "ZE_AFFINITY_MASK", "ZE_ENABLE_PCI_ID_DEVICE_ORDER"
    };
    
    std::stringstream env_info;
    bool first = true;
    
    for (const auto& var : env_vars) {
        const char* value = std::getenv(var.c_str());
        if (value) {
            if (!first) env_info << ",";
            env_info << var << "=" << value;
            first = false;
        }
    }
    
    return env_info.str().empty() ? "default" : env_info.str();
}
```

**Environment Variables Tracked:**
- `CCL_LOG_LEVEL`: OneCCL debugging level
- `NCCL_DEBUG`: NCCL debugging level  
- `OMP_NUM_THREADS`: OpenMP threading configuration
- `ZE_AFFINITY_MASK`: Intel GPU device affinity
- `ZE_ENABLE_PCI_ID_DEVICE_ORDER`: Intel GPU device ordering

### Goodput Implementation

#### Worst-Rank Timing Analysis
```cpp
static double calculate_goodput(const std::vector<double>& rank_times) {
    if (rank_times.empty()) return 0.0;
    return *std::max_element(rank_times.begin(), rank_times.end());
}
```

**Goodput Concept:**
- **Definition**: Performance of the slowest rank (worst-case timing)
- **Rationale**: Collective operations complete when the slowest rank finishes
- **Significance**: More realistic than average performance for collective operations

#### Usage in Summary Logging
```cpp
void log_summary(const std::string& data_type, size_t message_size_elements, 
                int num_ranks, double min_time_ms, double max_time_ms, double avg_time_ms) {
    std::cout << "Min Time: " << min_time_ms << " ms" << std::endl;
    std::cout << "Max Time: " << max_time_ms << " ms" << std::endl;
    std::cout << "Avg Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Goodput (worst-rank): " << max_time_ms << " ms" << std::endl;
}
```

## 📊 Data Format Evolution

### CSV Schema Evolution

#### Legacy Format (backward compatible)
```csv
timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,time_ms
```

#### Enhanced Format
```csv
timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment
```

#### Backward Compatibility Strategy
```python
def ensure_compatibility(df):
    """Add missing columns for backward compatibility."""
    if 'run_number' not in df.columns:
        df['run_number'] = range(1, len(df) + 1)
    
    if 'environment' not in df.columns:
        df['environment'] = 'unknown'
    
    return df
```

### Data Processing Pipeline

```
Raw CSV Data → Compatibility Check → Analysis → Visualization
     ↓              ↓                   ↓           ↓
Multiple files → Column validation → Statistics → Plots/Reports
```

## 🔬 Analysis Architecture

### Multi-Script Analysis System

#### Quick Analysis (`analyze_results.py`)
- **Purpose**: Fast overview of benchmark results
- **Features**: Basic statistics, best performance identification
- **Output**: Console summary with key metrics

#### Standard Visualization (`generate_plots.py`)
- **Purpose**: Comprehensive performance visualization
- **Features**: Message size analysis, data type comparison, scaling
- **Output**: PNG plots with performance trends

#### Advanced Goodput Analysis (`generate_goodput_plots.py`)
- **Purpose**: Deep dive into worst-rank timing analysis
- **Features**: Goodput trends, consistency analysis, environment impact
- **Output**: Detailed goodput-focused reports and visualizations

### Analysis Flow Architecture

```
CSV Files
    ↓
Load & Validate Data
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Quick Analysis  │ Standard Plots  │ Goodput Focus   │
│                 │                 │                 │
│ • Basic stats   │ • Message size  │ • Worst-rank    │
│ • Best perf     │ • Data types    │ • Consistency   │
│ • Overview      │ • Scaling       │ • Environment   │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                   ↓                   ↓
Console Output     Standard Plots    Goodput Reports
```

## 🔧 GPU Topology Investigation

### Intel GPU Max Analysis

#### Level-Zero Device Enumeration
```bash
#!/bin/bash
# investigate_gpu_topology.sh

echo "=== Intel GPU Max Topology Investigation ==="

# Check Level-Zero devices
if command -v level_zero_info &> /dev/null; then
    echo "=== Level-Zero Device Information ==="
    level_zero_info
else
    echo "level_zero_info not available, checking alternatives..."
fi

# Check GPU compute units
if command -v ocloc &> /dev/null; then
    echo "=== GPU Architecture Information ==="
    ocloc query -device pvc
else
    echo "ocloc not available"
fi
```

#### Xe Link Connectivity Analysis
```bash
# Check PCI topology
echo "=== PCI Device Topology ==="
lspci | grep -i display

# Check NUMA topology
echo "=== NUMA Topology ==="
numactl --hardware

# Check Intel GPU Max specific info
echo "=== Intel GPU Max Modules ==="
ls -la /sys/class/drm/card*/device/ | grep -E "(tile|memory)"
```

### Environment Optimization Guidelines

#### Intel GPU Max Optimization
```bash
# Device ordering and affinity
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="0.0,0.1"  # Specific GPU tiles

# Memory management
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="0"  # Use first GPU
```

## 🚀 Performance Optimization Architecture

### Benchmark Execution Optimization

#### MPI Process Distribution
```cpp
// Optimal rank-to-GPU mapping
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Set device based on rank for local node
int local_rank = rank % gpus_per_node;
set_device(local_rank);
```

#### Memory Management Strategy
```cpp
// Pre-allocate buffers for consistent timing
template<typename T>
void allocate_buffers(T** send_buf, T** recv_buf, size_t count) {
    // Allocate with proper alignment
    *send_buf = aligned_alloc(64, count * sizeof(T));
    *recv_buf = aligned_alloc(64, count * sizeof(T));
    
    // Initialize data to avoid first-touch penalties
    memset(*send_buf, 1, count * sizeof(T));
    memset(*recv_buf, 0, count * sizeof(T));
}
```

### Timing Precision Architecture

#### High-Resolution Timing
```cpp
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }
};
```

#### Synchronization Strategy
```cpp
// Ensure all ranks start simultaneously
MPI_Barrier(MPI_COMM_WORLD);

// Start timing
auto start = std::chrono::high_resolution_clock::now();

// Execute collective operation
collective_operation();

// Stop timing immediately
auto end = std::chrono::high_resolution_clock::now();

// Calculate duration
auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(
    end - start).count() / 1000.0;
```

## 🔄 Multi-Run Consistency Architecture

### Run-to-Run Variation Tracking

#### Statistical Analysis
```python
def analyze_run_consistency(df):
    """Analyze consistency across multiple runs."""
    grouped = df.groupby(['message_size_elements', 'data_type'])
    
    consistency_metrics = {}
    for (size, dtype), group in grouped:
        if len(group) > 1:
            times = group['time_ms']
            consistency_metrics[(size, dtype)] = {
                'mean': times.mean(),
                'std': times.std(),
                'cv': times.std() / times.mean(),  # Coefficient of variation
                'min': times.min(),
                'max': times.max(),
                'range_pct': (times.max() - times.min()) / times.mean() * 100
            }
    
    return consistency_metrics
```

#### Environment Impact Analysis
```python
def analyze_environment_impact(df):
    """Analyze how environment variables affect performance."""
    if 'environment' not in df.columns:
        return {}
    
    env_performance = df.groupby('environment')['time_ms'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(3)
    
    return env_performance
```

## 🧪 Testing Architecture

### Unit Testing Strategy
```cpp
// Test run counter functionality
void test_run_counter() {
    Logger::reset_global_run_counter();
    assert(Logger::get_current_run_id() == 0);
    
    Logger::start_new_run();
    assert(Logger::get_current_run_id() == 1);
    
    Logger::start_new_run();
    assert(Logger::get_current_run_id() == 2);
}

// Test environment capture
void test_environment_capture() {
    setenv("CCL_LOG_LEVEL", "trace", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    
    Logger logger("test", "lib", "op");
    std::string env = logger.capture_environment();
    
    assert(env.find("CCL_LOG_LEVEL=trace") != std::string::npos);
    assert(env.find("OMP_NUM_THREADS=1") != std::string::npos);
}
```

### Integration Testing
```bash
#!/bin/bash
# Integration test script

# Test backward compatibility
echo "Testing backward compatibility..."
python3 scripts/analyze_results.py --input test_data/legacy_csv/

# Test new functionality
echo "Testing new analysis features..."
python3 scripts/generate_goodput_plots.py --input test_data/enhanced_csv/

# Test multi-run analysis
echo "Testing multi-run consistency..."
./run_goodput_benchmark.sh --test-mode
```

## 📈 Scalability Considerations

### Multi-Node Architecture
```
Node 0          Node 1          Node N
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Rank 0  │    │ Rank 2  │    │ Rank N  │
│ Rank 1  │    │ Rank 3  │    │ Rank N+1│
│ Logger  │    │ Logger  │    │ Logger  │
│ Run ID  │────│ Run ID  │────│ Run ID  │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └─────────── Shared Run Counter ────────────┘
```

### Large-Scale Data Management
```python
def handle_large_datasets(results_dir, chunk_size=10000):
    """Process large CSV files in chunks."""
    all_files = glob.glob(f"{results_dir}/*.csv")
    
    for file_path in all_files:
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk in chunk_iter:
            # Process chunk
            processed_chunk = process_data_chunk(chunk)
            yield processed_chunk
```

## 🔮 Future Architecture Enhancements

### Real-Time Monitoring Integration
```python
# Prometheus metrics integration
from prometheus_client import Counter, Histogram, Gauge

benchmark_duration = Histogram(
    'collective_benchmark_duration_seconds',
    'Time spent in collective operation',
    ['library', 'collective', 'data_type', 'message_size']
)

goodput_gauge = Gauge(
    'collective_goodput_ms',
    'Goodput (worst-rank timing) for collective operation',
    ['library', 'collective', 'data_type']
)
```

### Machine Learning Integration
```python
# Performance prediction model
class PerformancePredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor()
        
    def train(self, features, targets):
        # Features: message_size, num_ranks, data_type, environment
        # Target: execution_time_ms
        self.model.fit(features, targets)
        
    def predict_performance(self, message_size, num_ranks, data_type, env):
        return self.model.predict([[message_size, num_ranks, data_type, env]])
```

### Advanced Analytics Pipeline
```
Real-time Data → Stream Processing → ML Models → Alerts/Predictions
      ↓               ↓                 ↓           ↓
CSV/Streaming → Apache Kafka → TensorFlow → Dashboard
```

## 📋 Development Guidelines

### Code Style
- **C++**: Follow Google C++ Style Guide
- **Python**: Follow PEP 8 with Black formatting
- **Shell**: Follow Google Shell Style Guide

### Adding New Analysis Features
1. **Extend Logger class**: Add new capture methods if needed
2. **Update CSV schema**: Add new columns (maintain backward compatibility)
3. **Update all analysis scripts**: Ensure compatibility across all tools
4. **Add tests**: Both unit and integration tests
5. **Update documentation**: Technical and user documentation

### Performance Considerations
- **Memory efficiency**: Use streaming for large datasets
- **CPU optimization**: Vectorized operations in analysis
- **I/O optimization**: Batch CSV operations
- **Caching**: Cache intermediate analysis results

This architecture provides a robust foundation for high-performance collective communication benchmarking with comprehensive analysis capabilities.
