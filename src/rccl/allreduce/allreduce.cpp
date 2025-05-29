#include <mpi.h>                           // MPI for multi-process
#include <rccl.h>                          // RCCL header
#include <hip/hip_runtime.h>               // HIP runtime API
#include <type_traits>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include "common/include/arg_parser.hpp"
#include "common/include/timer.hpp"
#include "common/include/logger.hpp"

using namespace std;

int main(int argc, char** argv) {
    // Parse arguments
    ArgParser parser(argc, argv);
    parser.add<string>("--dtype").add<size_t>("--count")
          .add<string>("--log").add<int>("--rank").add<int>("--world");
    parser.parse();

    // Extract parameters
    string dtype    = parser.get<string>("--dtype");
    size_t count    = parser.get<size_t>("--count");
    string logPath  = parser.get<string>("--log");
    int rank        = parser.get<int>("--rank");
    int world       = parser.get<int>("--world");

    // Ensure log directory
    filesystem::path p(logPath);
    if (auto dir = p.parent_path(); !dir.empty() && !filesystem::exists(dir))
        filesystem::create_directories(dir);

    // Initialize MPI and HIP device
    MPI_Init(NULL,NULL);
    hipError_t herr = hipSetDevice(rank);
    if (herr != hipSuccess)
        throw runtime_error(string("HIP error: ") + hipGetErrorString(herr));

    // Setup RCCL communicator
    rcclUniqueId id;
    if (rank==0) {
        rcclResult_t r = rcclGetUniqueId(&id);
        if (r!=rcclSuccess)
            throw runtime_error(string("RCCL error GetUniqueId: ") + rcclGetErrorString(r));
    }
    MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
    rcclComm_t comm;
    rcclResult_t rcr = rcclCommInitRank(&comm, world, id, rank);
    if (rcr!=rcclSuccess)
        throw runtime_error(string("RCCL error InitRank: ") + rcclGetErrorString(rcr));

    // Benchmark lambda
    auto bench = [&](auto tag) {
        using T = typename decltype(tag)::type;
        vector<T> h(count, T(rank));
        T *d_s, *d_r;
        // Allocate GPU memory
        if (hipMalloc(&d_s, count*sizeof(T)) != hipSuccess)
            throw runtime_error("HIP malloc failed");
        if (hipMalloc(&d_r, count*sizeof(T)) != hipSuccess)
            throw runtime_error("HIP malloc failed");
        // Copy to device
        if (hipMemcpy(d_s, h.data(), count*sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
            throw runtime_error("HIP memcpy failed");

        // Time and run allreduce
        Timer timer;
        rcclResult_t ar = rcclAllReduce(
            d_s, d_r, count,
            is_same<T,float>::value  ? rcclFloat :
            is_same<T,double>::value ? rcclDouble : rcclInt,
            rcclSum, comm, 0
        );
        if (ar!=rcclSuccess)
            throw runtime_error(string("RCCL error AllReduce: ") + rcclGetErrorString(ar));
        if (hipStreamSynchronize(0)!=hipSuccess)
            throw runtime_error("HIP stream sync failed");

        // Log time
        double ms = timer.elapsed_ms();
        Logger::append(logPath, {"rccl","allreduce",typeid(T).name(),to_string(count),to_string(ms)});

        // Free memory
        hipFree(d_s); hipFree(d_r);
    };

    // Dispatch data type
    if      (dtype=="float")  bench(type_tag<float>{});
    else if (dtype=="double") bench(type_tag<double>{});
    else if (dtype=="int")    bench(type_tag<int>{});
    else throw runtime_error("Unsupported dtype: " + dtype);

    // Cleanup
    rcclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}