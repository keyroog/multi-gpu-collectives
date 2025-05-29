#include <mpi.h>                           // MPI for launcher
#include <nccl.h>                          // NCCL header
#include <cuda_runtime.h>                  // CUDA runtime API
#include <type_traits>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include "common/include/arg_parser.hpp"  // CLI parsing
#include "common/include/timer.hpp"       // RAII timer
#include "common/include/logger.hpp"      // CSV logger

using namespace std;

int main(int argc, char** argv) {
    // 1) Parse arguments
    ArgParser parser(argc, argv);
    parser.add<string>("--dtype").add<size_t>("--count")
          .add<string>("--log").add<int>("--rank").add<int>("--world");
    parser.parse();

    // 2) Extract parameters
    string dtype    = parser.get<string>("--dtype");
    size_t count    = parser.get<size_t>("--count");
    string logPath  = parser.get<string>("--log");
    int rank        = parser.get<int>("--rank");
    int world       = parser.get<int>("--world");

    // Ensure log directory exists
    filesystem::path p(logPath);
    if (auto dir = p.parent_path(); !dir.empty() && !filesystem::exists(dir))
        filesystem::create_directories(dir);

    // 3) Initialize MPI and CUDA
    MPI_Init(NULL, NULL);
    cudaError_t cerr = cudaSetDevice(rank);
    if (cerr != cudaSuccess)
        throw runtime_error(string("CUDA error: ") + cudaGetErrorString(cerr));

    // 4) Setup NCCL communicator
    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t res = ncclGetUniqueId(&id);
        if (res != ncclSuccess)
            throw runtime_error(string("NCCL error GetUniqueId: ") + ncclGetErrorString(res));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclResult_t cres = ncclCommInitRank(&comm, world, id, rank);
    if (cres != ncclSuccess)
        throw runtime_error(string("NCCL error CommInitRank: ") + ncclGetErrorString(cres));

    // 5) Benchmark lambda
    auto bench = [&](auto tag) {
        using T = typename decltype(tag)::type;
        vector<T> h(count, T(rank));
        T *d_s, *d_r;
        // Allocate GPU buffers
        if (cudaMalloc(&d_s, count*sizeof(T)) != cudaSuccess)
            throw runtime_error("CUDA malloc failed");
        if (cudaMalloc(&d_r, count*sizeof(T)) != cudaSuccess)
            throw runtime_error("CUDA malloc failed");
        // Transfer to GPU
        if (cudaMemcpy(d_s, h.data(), count*sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
            throw runtime_error("CUDA memcpy failed");

        Timer timer;
        ncclResult_t ar = ncclAllReduce(
            d_s, d_r, count,
            is_same<T,float>::value  ? ncclFloat :
            is_same<T,double>::value ? ncclDouble : ncclInt,
            ncclSum, comm, 0
        );
        if (ar != ncclSuccess)
            throw runtime_error(string("NCCL error AllReduce: ") + ncclGetErrorString(ar));
        if (cudaStreamSynchronize(0) != cudaSuccess)
            throw runtime_error("CUDA stream sync failed");

        // Log time
        double ms = timer.elapsed_ms();
        Logger::append(logPath, {"nccl","allreduce",typeid(T).name(),to_string(count),to_string(ms)});

        // Cleanup
        cudaFree(d_s);
        cudaFree(d_r);
    };

    // 6) Dispatch and run
    if      (dtype=="float")  bench(type_tag<float>{});
    else if (dtype=="double") bench(type_tag<double>{});
    else if (dtype=="int")    bench(type_tag<int>{});
    else throw runtime_error("Unsupported dtype: " + dtype);

    // Finalize NCCL and MPI
    if (ncclCommDestroy(comm) != ncclSuccess)
        throw runtime_error("NCCL error CommDestroy");
    MPI_Finalize();
    return 0;
}