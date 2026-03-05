// filepath: src/mpi/alltoall/alltoall.cu
#include "../common/mpi_context.hpp"
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <string>

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
MPI_Datatype mpi_dtype_of();
template<> MPI_Datatype mpi_dtype_of<int>()    { return MPI_INT; }
template<> MPI_Datatype mpi_dtype_of<float>()  { return MPI_FLOAT; }
template<> MPI_Datatype mpi_dtype_of<double>() { return MPI_DOUBLE; }

template<typename T>
void run_alltoall(size_t count_per_dest, size_t global_count, int size, int rank, MpiContext& ctx, const std::string& data_type) {
    T* send_buf;
    T* recv_buf;
    cudaMalloc(&send_buf, count_per_dest * size * sizeof(T));
    cudaMalloc(&recv_buf, count_per_dest * size * sizeof(T));

    int threads = 256;
    int blocks = (count_per_dest * size + threads - 1) / threads;
    init_buffers<T><<<blocks, threads, 0, ctx.stream>>>(send_buf, recv_buf, count_per_dest, rank, size);
    cudaStreamSynchronize(ctx.stream);

    MPI_Datatype dtype = mpi_dtype_of<T>();

    // warmup
    cudaStreamSynchronize(ctx.stream);
    MPI_Alltoall(send_buf, count_per_dest, dtype, recv_buf, count_per_dest, dtype, MPI_COMM_WORLD);

    // timed run
    cudaStreamSynchronize(ctx.stream);
    double t_start = MPI_Wtime();
    MPI_Alltoall(send_buf, count_per_dest, dtype, recv_buf, count_per_dest, dtype, MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed_ms = (t_end - t_start) * 1000.0;
    std::cout << "Rank " << rank << " alltoall time: "
              << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";

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
    ctx.logger.log_result(data_type, global_count, size, rank, ok, ctx.init_time_ms, elapsed_ms);
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
    try { output_dir = parser.get<std::string>("--output"); } catch (...) { output_dir = ""; }
    if (global_count == 0) global_count = 1024 * 1024;

    auto ctx = init_mpi(output_dir, "alltoall", argc, argv);
    int size = ctx.size;
    int rank = ctx.rank;

    size_t denom = static_cast<size_t>(size) * static_cast<size_t>(size);
    size_t count_per_dest = global_count / denom;
    size_t remainder      = global_count % denom;
    size_t effective_global = count_per_dest * denom;
    if (count_per_dest == 0) {
        if (rank == 0) std::cerr << "Global count too small for size=" << size << " (needs at least size^2 elements).\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (rank == 0 && remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") is not divisible by size^2 (" << denom
                  << "). Using count_per_dest=" << count_per_dest
                  << " and ignoring last " << remainder << " elements.\n";
    }

    if      (dtype == "int")    run_alltoall<int>   (count_per_dest, effective_global, size, rank, ctx, dtype);
    else if (dtype == "float")  run_alltoall<float> (count_per_dest, effective_global, size, rank, ctx, dtype);
    else if (dtype == "double") run_alltoall<double>(count_per_dest, effective_global, size, rank, ctx, dtype);
    else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    finalize_mpi(ctx);
    return 0;
}
