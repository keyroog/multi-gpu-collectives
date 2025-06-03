# Makefile per OneCCL Collectives Benchmarking
# Requires Intel oneAPI toolkit

# Compiler and flags
CXX = icpx
CXXFLAGS = -fsycl -std=c++17 -O3 -Wall
INCLUDES = -I./src/common/include
LIBS = -lmpi -lccl

# Common source files
COMMON_SRC = src/common/src/oneccl_setup.cpp
COMMON_HEADERS = $(wildcard src/common/include/*.hpp)

# Output directories
BUILD_DIR = build
BIN_DIR = bin

# Targets
TARGETS = allreduce allgather

# Default target
all: setup $(TARGETS)

# Create directories
setup:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# AllReduce benchmark
allreduce: $(BIN_DIR)/allreduce_benchmark

$(BIN_DIR)/allreduce_benchmark: src/oneccl/allreduce/allreduce.cpp $(COMMON_SRC) $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ src/oneccl/allreduce/allreduce.cpp $(COMMON_SRC) $(LIBS)

# AllGather benchmark
allgather: $(BIN_DIR)/allgather_benchmark

$(BIN_DIR)/allgather_benchmark: src/oneccl/allgather/allgather.cpp $(COMMON_SRC) $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ src/oneccl/allgather/allgather.cpp $(COMMON_SRC) $(LIBS)

# Test compilation (no linking)
test-compile:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/oneccl/allreduce/allreduce.cpp -o $(BUILD_DIR)/allreduce.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(COMMON_SRC) -o $(BUILD_DIR)/oneccl_setup.o
	@echo "Compilation test passed!"

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Example runs (requires MPI environment)
run-allreduce: $(BIN_DIR)/allreduce_benchmark
	mpirun -n 2 $(BIN_DIR)/allreduce_benchmark --dtype float --count 1000 --output ./results

run-allgather: $(BIN_DIR)/allgather_benchmark
	mpirun -n 2 $(BIN_DIR)/allgather_benchmark --dtype int --count 1000 --output ./results

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all benchmarks"
	@echo "  allreduce    - Build AllReduce benchmark"
	@echo "  allgather    - Build AllGather benchmark"
	@echo "  test-compile - Test compilation without linking"
	@echo "  clean        - Clean build files"
	@echo "  run-allreduce - Run AllReduce benchmark example"
	@echo "  run-allgather - Run AllGather benchmark example"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Intel oneAPI toolkit"
	@echo "  - Source oneAPI environment: source /opt/intel/oneapi/setvars.sh"
	@echo "  - MPI environment for running benchmarks"

.PHONY: all setup clean test-compile run-allreduce run-allgather help
