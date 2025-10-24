// filepath: src/nccl/scatter/scatter.cu
#include "../common/nccl_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Kernel to initialize device buffers for scatter
template<typename T>
__global__ void init_buffers(T* send_buf, T* recv_buf, size_t count, int size, int rank, int root) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank == root) {
        size_t total = count * size;
        if (id < total) {
            int dest = id / count;
            size_t idx = id % count;
            // data pattern: root sends (root*1000 + dest*100 + idx)
            send_buf[id] = static_cast<T>(root * 1000 + dest * 100 + idx);
        }
    }
    if (id < count) {
        recv_buf[id] = static_cast<T>(-1);
    }
}

// Run NCCL scatter using point-to-point sends/receives
template<typename T>
void run_scatter(size_t count, int size, int rank, NcclContext& ctx, const std::string& data_type) {
    int root = 0;
    // select nccl dtype
    ncclDataType_t nccl_dtype = (data_type == "float" ? ncclFloat
                                : data_type == "double" ? ncclDouble
                                : ncclInt);

    // allocate device buffers
    T* send_buf; cudaMalloc(&send_buf, count * size * sizeof(T));
    T* recv_buf; cudaMalloc(&recv_buf, count * sizeof(T));

    // init buffers
    int threads = 256;
    int blocks = ((count * size) + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, count, size, rank, root);
    cudaStreamSynchronize(ctx.stream);

    // warm-up non misurata
    ncclGroupStart();
    if (rank == root) {
        for (int peer = 0; peer < size; ++peer) {
            ncclSend(send_buf + peer * count, count, nccl_dtype, peer, ctx.comm, ctx.stream);
        }
    }
    ncclRecv(recv_buf, count, nccl_dtype, root, ctx.comm, ctx.stream);
    ncclGroupEnd();
    cudaStreamSynchronize(ctx.stream);

    // perform scatter and time it 5 times
    for (int iter = 0; iter < 5; ++iter) {
        auto t_start = std::chrono::high_resolution_clock::now();
        ncclGroupStart();
        if (rank == root) {
            for (int peer = 0; peer < size; ++peer) {
                ncclSend(send_buf + peer * count, count, nccl_dtype, peer, ctx.comm, ctx.stream);
            }
        }
        // all ranks receive their chunk
        ncclRecv(recv_buf, count, nccl_dtype, root, ctx.comm, ctx.stream);
        ncclGroupEnd();
        cudaStreamSynchronize(ctx.stream);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;

        // log and print
        ctx.logger.log_result(data_type, count, size, rank, elapsed_ms);
        std::cout << "Rank " << rank << " scatter time (iter " << iter << "): "
                  << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    }
    
    // correctness check
    T* host_buf = new T[count];
    cudaMemcpy(host_buf, recv_buf, count * sizeof(T), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(root * 1000 + rank * 100 + i);
        if (host_buf[i] != expected) { ok = false; break; }
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
    size_t count = parser.get<size_t>("--count");
    std::string output_dir;
    try { output_dir = parser.get<std::string>("--output"); } catch(...) { output_dir = ""; }
    if (count == 0) count = 1024 * 1024;

    auto ctx = init_nccl(output_dir, "scatter", argc, argv);
    int size = ctx.size;
    int rank = ctx.rank;

    if (dtype == "int") run_scatter<int>(count, size, rank, ctx, dtype);
    else if (dtype == "float") run_scatter<float>(count, size, rank, ctx, dtype);
    else if (dtype == "double") run_scatter<double>(count, size, rank, ctx, dtype);
    else { std::cerr << "Unsupported dtype: " << dtype << std::endl; return -1; }
    return 0;
}
