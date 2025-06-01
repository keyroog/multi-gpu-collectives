#!/bin/bash

# Script per investigare la topologia GPU Intel Max
# Analizza la configurazione hardware per ottimizzare le performance collective

set -e

echo "=== INTEL GPU MAX TOPOLOGY INVESTIGATION ==="
echo "Date: $(date)"
echo "System: $(uname -a)"
echo

# Funzione per controllare se un comando esiste
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Informazioni GPU Level-Zero
echo "=== 1. LEVEL-ZERO GPU INFORMATION ==="
if command_exists level_zero_info; then
    echo "Running level_zero_info..."
    level_zero_info
else
    echo "⚠️  level_zero_info not found. Install Intel GPU drivers/tools."
fi
echo

# 2. Intel OpenCL Compiler Query
echo "=== 2. INTEL OCLOC DEVICE QUERY ==="
if command_exists ocloc; then
    echo "Available GPU devices:"
    ocloc query -list_devices
    echo
    echo "Detailed device capabilities:"
    ocloc query -capabilities
else
    echo "⚠️  ocloc not found. Install Intel OpenCL tools."
fi
echo

# 3. Intel OneAPI Environment
echo "=== 3. ONEAPI ENVIRONMENT ==="
if [ -n "$ONEAPI_ROOT" ]; then
    echo "OneAPI Root: $ONEAPI_ROOT"
    echo "OneAPI Version: $(cat $ONEAPI_ROOT/version.txt 2>/dev/null || echo 'Unknown')"
else
    echo "⚠️  OneAPI environment not loaded. Run: source /opt/intel/oneapi/setvars.sh"
fi

# Variabili d'ambiente importanti
echo "Environment variables:"
echo "  CCL_ROOT: ${CCL_ROOT:-'Not set'}"
echo "  LEVEL_ZERO_ROOT: ${LEVEL_ZERO_ROOT:-'Not set'}"
echo "  ZE_DEBUG: ${ZE_DEBUG:-'Not set'}"
echo "  ZE_ENABLE_PCI_ID_DEVICE_ORDER: ${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-'Not set'}"
echo

# 4. System GPU Information
echo "=== 4. SYSTEM GPU DETECTION ==="
if command_exists lspci; then
    echo "PCI GPU devices:"
    lspci | grep -i "vga\|3d\|display\|gpu" || echo "No GPU devices found via lspci"
else
    echo "lspci not available on macOS"
fi

# Per macOS, usa system_profiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS Graphics information:"
    system_profiler SPDisplaysDataType | grep -E "(Chipset Model|VRAM|Metal)" || echo "Graphics info not available"
fi
echo

# 5. Memory e NUMA topology (Linux only)
echo "=== 5. MEMORY AND NUMA TOPOLOGY ==="
if [[ "$OSTYPE" == "linux"* ]]; then
    if command_exists numactl; then
        echo "NUMA topology:"
        numactl --hardware
    else
        echo "numactl not found"
    fi
    
    if [ -f /proc/meminfo ]; then
        echo "Memory information:"
        grep -E "(MemTotal|MemAvailable)" /proc/meminfo
    fi
else
    echo "Memory information (non-Linux):"
    if command_exists free; then
        free -h
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Total Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
    fi
fi
echo

# 6. Test Level-Zero device enumeration
echo "=== 6. LEVEL-ZERO DEVICE ENUMERATION TEST ==="
cat > /tmp/ze_device_test.cpp << 'EOF'
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>

int main() {
    ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        std::cout << "Failed to initialize Level-Zero" << std::endl;
        return 1;
    }
    
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);
    std::cout << "Number of Level-Zero drivers: " << driverCount << std::endl;
    
    if (driverCount == 0) {
        std::cout << "No Level-Zero drivers found" << std::endl;
        return 1;
    }
    
    std::vector<ze_driver_handle_t> drivers(driverCount);
    zeDriverGet(&driverCount, drivers.data());
    
    for (uint32_t i = 0; i < driverCount; ++i) {
        uint32_t deviceCount = 0;
        zeDeviceGet(drivers[i], &deviceCount, nullptr);
        std::cout << "Driver " << i << " has " << deviceCount << " devices" << std::endl;
        
        if (deviceCount > 0) {
            std::vector<ze_device_handle_t> devices(deviceCount);
            zeDeviceGet(drivers[i], &deviceCount, devices.data());
            
            for (uint32_t j = 0; j < deviceCount; ++j) {
                ze_device_properties_t deviceProps = {};
                deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                zeDeviceGetProperties(devices[j], &deviceProps);
                
                std::cout << "  Device " << j << ": " << deviceProps.name << std::endl;
                std::cout << "    Type: " << (deviceProps.type == ZE_DEVICE_TYPE_GPU ? "GPU" : "Other") << std::endl;
                std::cout << "    Max memory: " << deviceProps.maxMemAllocSize / (1024*1024) << " MB" << std::endl;
                std::cout << "    Max compute units: " << deviceProps.numEUsPerSubslice * deviceProps.numSubslicesPerSlice * deviceProps.numSlices << std::endl;
            }
        }
    }
    
    return 0;
}
EOF

# Prova a compilare e eseguire il test Level-Zero
if command_exists icpx && [ -n "$LEVEL_ZERO_ROOT" ]; then
    echo "Compiling Level-Zero device test..."
    if icpx -o /tmp/ze_device_test /tmp/ze_device_test.cpp -lze_loader 2>/dev/null; then
        echo "Running Level-Zero device enumeration:"
        /tmp/ze_device_test
        rm -f /tmp/ze_device_test
    else
        echo "⚠️  Failed to compile Level-Zero test"
    fi
else
    echo "⚠️  icpx or Level-Zero not available for device test"
fi
rm -f /tmp/ze_device_test.cpp
echo

# 7. Raccomandazioni per topology optimization
echo "=== 7. OPTIMIZATION RECOMMENDATIONS ==="
echo "For Intel GPU Max systems:"
echo "  1. Check if multiple GPUs are on same 'MACRO GPU' package"
echo "  2. Verify Xe Link connectivity between GPU tiles"
echo "  3. Ensure proper NUMA affinity for multi-socket systems"
echo "  4. Use ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 for consistent device ordering"
echo "  5. Monitor CCL_LOG_LEVEL=trace for OneCCL device selection"
echo
echo "Environment variables for optimal performance:"
echo "  export ZE_DEBUG=1                              # Level-Zero debug info"
echo "  export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1        # Consistent device order"
echo "  export CCL_LOG_LEVEL=warn                      # OneCCL logging level"
echo "  export OMP_NUM_THREADS=1                       # Control OpenMP threading"
echo
echo "For debugging performance issues:"
echo "  export CCL_LOG_LEVEL=trace                     # Detailed OneCCL logs"
echo "  export NCCL_DEBUG=TRACE                        # NCCL debug (if applicable)"
echo "  export ZE_DEBUG=1                              # Level-Zero debug"
echo

echo "=== INVESTIGATION COMPLETE ==="
echo "Save this output for performance analysis and tuning."
