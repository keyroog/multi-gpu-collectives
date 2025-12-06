// filepath: src/nccl/reduce_scatter/reduce_scatter.cu
#include "../common/nccl_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Kernel to initialize device buffers for reduce_scatter
template<typename T>
__global__ void init_buffers(T* send_buf, T* recv_buf, size_t count, int size, int rank) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * size;
    if (id < total) {
        // Data pattern: unique per rank
        send_buf[id] = static_cast<T>((rank + 1) * (id + 1));
    }
    if (id < count) {
        // initialize receive buffer to sentinel
        recv_buf[id] = static_cast<T>(-1);
    }
}

template<typename T>
void run_reduce_scatter(size_t local_count, size_t global_count, int size, int rank, NcclContext& ctx, const std::string& data_type) {
    // determine NCCL data type
    ncclDataType_t nccl_dtype;
    if (data_type == "int") nccl_dtype = ncclInt;
    else if (data_type == "float") nccl_dtype = ncclFloat;
    else /* double */ nccl_dtype = ncclDouble;

    // allocate device buffers
    T* send_buf;
    T* recv_buf;
    cudaMalloc(&send_buf, local_count * size * sizeof(T));
    cudaMalloc(&recv_buf, local_count * sizeof(T));

    // initialize buffers
    int threads = 256;
    int blocks = (local_count * size + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, local_count, size, rank);
    cudaStreamSynchronize(ctx.stream);

    // perform reduce_scatter and time it once
    auto t_start = std::chrono::high_resolution_clock::now();
    ncclReduceScatter(send_buf, recv_buf, local_count, nccl_dtype, ncclSum, ctx.comm, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    std::cout << "Rank " << rank << " reduce_scatter time: "
              << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";

    // correctness check
    T* host_buf = new T[local_count];
    cudaMemcpy(host_buf, recv_buf, local_count * sizeof(T), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < local_count && ok; ++i) {
        size_t global_idx = rank * local_count + i;
        T expected = static_cast<T>(0);
        for (int r = 0; r < size; ++r) {
            expected += static_cast<T>((r + 1) * (global_idx + 1));
        }
        if (host_buf[i] != expected) {
            ok = false;
        }
    }
    std::cout << (ok ? "PASSED\n" : "FAILED\n");
    ctx.logger.log_result(data_type, global_count, size, rank, ok, elapsed_ms);
    delete[] host_buf;

    // cleanup
    cudaFree(send_buf);
    cudaFree(recv_buf);
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count");
    parser.parse();

    std::string dtype = parser.get<std::string>("--dtype");
    size_t global_count = parser.get<size_t>("--count");
    std::string output_dir;
    try {
        output_dir = parser.get<std::string>("--output");
    } catch (...) {
        output_dir = "";
    }
    if (global_count == 0) global_count = 1024 * 1024;

    // Initialize NCCL context for reduce_scatter
    auto ctx = init_nccl(output_dir, "reduce_scatter", argc, argv);
    int size = ctx.size;
    int rank = ctx.rank;

    size_t local_count = global_count / size;
    size_t remainder   = global_count % size;
    size_t effective_global = local_count * size;
    if (local_count == 0) {
        if (rank == 0) {
            std::cerr << "Global count too small for size=" << size << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (rank == 0 && remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") is not divisible by size (" << size
                  << "). Using local_count=" << local_count
                  << " and ignoring last " << remainder << " elements.\n";
    }

    // dispatch based on data type
    if (dtype == "int") {
        run_reduce_scatter<int>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "float") {
        run_reduce_scatter<float>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "double") {
        run_reduce_scatter<double>(local_count, effective_global, size, rank, ctx, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return 0;
}
