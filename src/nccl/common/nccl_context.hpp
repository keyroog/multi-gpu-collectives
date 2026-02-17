#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "../../common/include/logger.hpp"

struct NcclContext {
    int size;
    int rank;
    int device;
    ncclComm_t comm;
    cudaStream_t stream;
    double init_time_ms;
    Logger logger;
};

inline NcclContext init_nccl(const std::string& output_dir, const std::string& collective_name, 
                             int argc = 0, char** argv = nullptr) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get local rank (position within the node) for GPU selection
    MPI_Comm local_comm;
    int local_rank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Select GPU device based on local rank
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices < 1 || local_rank >= nDevices) {
        std::cerr << "Rank " << rank << " (local " << local_rank
                  << "): not enough GPUs (found " << nDevices << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int device = local_rank;
    cudaSetDevice(device);
    
    // Create NCCL communicator
    double t_init_start = MPI_Wtime();
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    double t_init_end = MPI_Wtime();
    double init_time_ms = (t_init_end - t_init_start) * 1000.0;
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
    
    // Initialize logger
    Logger logger(output_dir, "nccl", collective_name);

    return NcclContext{size, rank, device, comm, stream, init_time_ms, logger};
}

inline void finalize_nccl(NcclContext& ctx) {
    // Sincronizza tutti i rank prima del cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Destroy NCCL communicator
    ncclCommDestroy(ctx.comm);

    // Destroy CUDA stream
    cudaStreamDestroy(ctx.stream);

    // Finalize MPI
    MPI_Finalize();
}
