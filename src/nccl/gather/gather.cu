// filepath: src/nccl/gather/gather.cu
#include "../common/nccl_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Kernel to initialize device buffers for gather
template<typename T>
__global__ void init_buffers(T* send_buf, T* recv_buf, size_t count, int size, int rank) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        // send buffer initialized uniquely per rank
        send_buf[id] = static_cast<T>(rank * 1000 + id);
        // recv buffer only needed on root
        for (int r = 0; r < size; ++r) {
            recv_buf[r * count + id] = static_cast<T>(-1);
        }
    }
}

template<typename T>
void run_gather(size_t local_count, size_t global_count, int size, int rank, NcclContext& ctx, const std::string& data_type) {
    int root = 0;
    // determine NCCL data type
    ncclDataType_t nccl_dtype = (data_type == "float" ? ncclFloat
                                : data_type == "double" ? ncclDouble
                                : ncclInt);

    // allocate device buffers
    T* send_buf; cudaMalloc(&send_buf, local_count * sizeof(T));
    T* recv_buf; cudaMalloc(&recv_buf, local_count * size * sizeof(T));

    // initialize buffers
    int threads = 256;
    int blocks = (local_count + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, local_count, size, rank);
    cudaStreamSynchronize(ctx.stream);

    // perform gather and time it once
    auto t_start = std::chrono::high_resolution_clock::now();
    ncclGroupStart();
    for (int peer = 0; peer < size; ++peer) {
        if (rank == peer) {
            ncclSend(send_buf, local_count, nccl_dtype, root, ctx.comm, ctx.stream);
        }
        if (rank == root) {
            ncclRecv(recv_buf + peer * local_count, local_count, nccl_dtype, peer, ctx.comm, ctx.stream);
        }
    }
    ncclGroupEnd();
    cudaStreamSynchronize(ctx.stream);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;

    // log result once per rank
    ctx.logger.log_result(data_type, global_count, size, rank, elapsed_ms);
    std::cout << "Rank " << rank << " gather time: "
              << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";

    // correctness check on root only
    if (rank == root) {
        T* host_buf = new T[local_count * size];
        cudaMemcpy(host_buf, recv_buf, local_count * size * sizeof(T), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int src = 0; src < size && ok; ++src) {
            for (size_t i = 0; i < local_count; ++i) {
                T expected = static_cast<T>(src * 1000 + i);
                if (host_buf[src * local_count + i] != expected) { ok = false; break; }
            }
        }
        std::cout << (ok ? "PASSED\n" : "FAILED\n");
        delete[] host_buf;
    }

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
    try { output_dir = parser.get<std::string>("--output"); } catch (...) { output_dir = ""; }
    if (global_count == 0) global_count = 1024 * 1024;

    // Initialize NCCL context for gather
    auto ctx = init_nccl(output_dir, "gather", argc, argv);
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
        run_gather<int>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "float") {
        run_gather<float>(local_count, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "double") {
        run_gather<double>(local_count, effective_global, size, rank, ctx, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return 0;
}
