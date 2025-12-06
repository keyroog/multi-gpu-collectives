// filepath: src/nccl/allreduce/allreduce.cpp
#include "../common/nccl_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Kernel to initialize device buffers
template<typename T>
__global__ void init_buffers(T* send_buf, T* recv_buf, size_t count, int rank) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        send_buf[id] = static_cast<T>(rank + id + 1);
        recv_buf[id] = static_cast<T>(-1);
    }
}

template<typename T>
void run_allreduce(size_t local_count, size_t global_count, int size, int rank, NcclContext& ctx, const std::string& data_type) {
    // determine NCCL data type
    ncclDataType_t nccl_dtype;
    if (data_type == "int") nccl_dtype = ncclInt;
    else if (data_type == "float") nccl_dtype = ncclFloat;
    else /* double */ nccl_dtype = ncclDouble;
    
    // allocate device buffers
    T* send_buf;
    T* recv_buf;
    cudaMalloc(&send_buf, local_count * sizeof(T));
    cudaMalloc(&recv_buf, local_count * sizeof(T));

    // initialize buffers
    int threads = 256;
    int blocks = (local_count + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, local_count, rank);
    cudaStreamSynchronize(ctx.stream);

    // compute expected sum
    T check_sum = static_cast<T>(0);
    for (int i = 1; i <= size; ++i) check_sum += static_cast<T>(i);

    // perform allreduce and time it once
    auto t_start = std::chrono::high_resolution_clock::now();
    ncclAllReduce(send_buf, recv_buf, local_count, nccl_dtype, ncclSum, ctx.comm, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    std::cout << "Rank " << rank << " allreduce time: "
              << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";

    // correctness check
    T* host_buf = new T[local_count];
    cudaMemcpy(host_buf, recv_buf, local_count * sizeof(T), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < local_count; ++i) {
        if (host_buf[i] != static_cast<T>(check_sum + size * i)) { ok = false; break; }
    }
    std::cout << (ok ? "PASSED\n" : "FAILED\n");
    ctx.logger.log_result(data_type, global_count, size, rank, ok, elapsed_ms);
    delete[] host_buf;

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
    if (global_count == 0) global_count = 10 * 1024 * 1024;

    // Initialize NCCL context
    auto ctx = init_nccl(output_dir, "allreduce", argc, argv);  
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

    // dispatch based on dtype
    if (dtype == "int") {
        run_allreduce<int>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "float") {
        run_allreduce<float>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "double") {
        run_allreduce<double>(local_count, effective_global, size, rank, ctx, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return 0;
}
