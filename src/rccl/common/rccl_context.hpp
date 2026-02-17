#pragma once

#include <mpi.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include "../../common/include/logger.hpp"

struct RcclContext {
    int size;
    int rank;
    int device;
    ncclComm_t comm;
    hipStream_t stream;
    double init_time_ms;
    Logger logger;
};

inline RcclContext init_rccl(const std::string& output_dir, const std::string& collective_name,
                             int argc = 0, char** argv = nullptr) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Select GPU device based on rank
    int nDevices = 0;
    hipGetDeviceCount(&nDevices);
    if (nDevices < size) {
        if (rank == 0) std::cerr << "Not enough GPUs for all ranks" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int device = rank % nDevices;
    hipSetDevice(device);

    // Create RCCL communicator
    double t_init_start = MPI_Wtime();
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    double t_init_end = MPI_Wtime();
    double init_time_ms = (t_init_end - t_init_start) * 1000.0;

    // Create HIP stream
    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamDefault);

    // Initialize logger
    Logger logger(output_dir, "rccl", collective_name);

    return RcclContext{size, rank, device, comm, stream, init_time_ms, logger};
}

inline void finalize_rccl(RcclContext& ctx) {
    // Sincronizza tutti i rank prima del cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Destroy RCCL communicator
    ncclCommDestroy(ctx.comm);

    // Destroy HIP stream
    hipStreamDestroy(ctx.stream);

    // Finalize MPI
    MPI_Finalize();
}
