// filepath: src/nccl/alltoall/alltoall.cu
#include "../common/nccl_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Kernel to initialize device buffers for alltoall
template<typename T>
__global__ void init_buffers(T* send_buf, T* recv_buf, size_t count, int rank, int size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * size;
    if (id < total) {
        int dest_rank = id / count;
        size_t local_id = id % count;
        send_buf[id] = static_cast<T>(rank * 1000 + dest_rank * 100 + local_id);
        recv_buf[id] = static_cast<T>(-1);
    }
}

template<typename T>
void run_alltoall(size_t count_per_dest, size_t global_count, int size, int rank, NcclContext& ctx, const std::string& data_type) {
    // determine NCCL data type
    ncclDataType_t nccl_dtype;
    if (data_type == "int") nccl_dtype = ncclInt;
    else if (data_type == "float") nccl_dtype = ncclFloat;
    else /* double */ nccl_dtype = ncclDouble;

    // allocate device buffers
    T* send_buf;
    T* recv_buf;
    cudaMalloc(&send_buf, count_per_dest * size * sizeof(T));
    cudaMalloc(&recv_buf, count_per_dest * size * sizeof(T));

    // initialize buffers
    int threads = 256;
    int blocks = (count_per_dest * size + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, count_per_dest, rank, size);
    cudaStreamSynchronize(ctx.stream);

    // warm-up non misurata
    ncclGroupStart();
    for (int peer = 0; peer < size; ++peer) {
        ncclRecv(recv_buf + peer * count_per_dest, count_per_dest, nccl_dtype, peer, ctx.comm, ctx.stream);
        ncclSend(send_buf + peer * count_per_dest, count_per_dest, nccl_dtype, peer, ctx.comm, ctx.stream);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(ctx.stream);

    // perform alltoall and time it 5 times
    for (int iter = 0; iter < 5; ++iter) {
        auto t_start = std::chrono::high_resolution_clock::now();
        ncclGroupStart();
        for (int peer = 0; peer < size; ++peer) {
            ncclRecv(recv_buf + peer * count_per_dest, count_per_dest, nccl_dtype, peer, ctx.comm, ctx.stream);
            ncclSend(send_buf + peer * count_per_dest, count_per_dest, nccl_dtype, peer, ctx.comm, ctx.stream);
        }
        ncclGroupEnd();
        cudaStreamSynchronize(ctx.stream);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
        ctx.logger.log_result(data_type, global_count, size, rank, elapsed_ms);
        std::cout << "Rank " << rank << " alltoall time (iter " << iter << "): "
                  << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    }

    // correctness check
    T* host_buf = new T[count_per_dest * size];
    cudaMemcpy(host_buf, recv_buf, count_per_dest * size * sizeof(T), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int src = 0; src < size && ok; ++src) {
        for (size_t i = 0; i < count_per_dest; ++i) {
            T expected = static_cast<T>(src * 1000 + rank * 100 + i);
            if (host_buf[src * count_per_dest + i] != expected) { ok = false; break; }
        }
    }
    std::cout << (ok ? "PASSED\n" : "FAILED\n");
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
    if (global_count == 0) global_count = 1024 * 1024;

    // Initialize NCCL context for alltoall
    auto ctx = init_nccl(output_dir, "alltoall", argc, argv);
    int size = ctx.size;
    int rank = ctx.rank;

    size_t denom = static_cast<size_t>(size) * static_cast<size_t>(size);
    size_t count_per_dest = global_count / denom;
    size_t remainder      = global_count % denom;
    size_t effective_global = count_per_dest * denom;
    if (count_per_dest == 0) {
        if (rank == 0) {
            std::cerr << "Global count too small for size=" << size << " (needs at least size^2 elements).\n";
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (rank == 0 && remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") is not divisible by size^2 (" << denom
                  << "). Using count_per_dest=" << count_per_dest
                  << " and ignoring last " << remainder << " elements.\n";
    }

    // dispatch based on data type
    if (dtype == "int") {
        run_alltoall<int>(count_per_dest, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "float") {
        run_alltoall<float>(count_per_dest, effective_global, size, rank, ctx, dtype);
    } else if (dtype == "double") {
        run_alltoall<double>(count_per_dest, effective_global, size, rank, ctx, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return 0;
}
