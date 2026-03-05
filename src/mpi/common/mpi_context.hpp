#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <chrono>
#include "../../common/include/logger.hpp"

struct MpiContext {
    int size;
    int rank;
    int device;
    cudaStream_t stream;
    double init_time_ms;
    Logger logger;
};

inline MpiContext init_mpi(const std::string& output_dir, const std::string& collective_name,
                            int argc = 0, char** argv = nullptr) {
    // Time MPI_Init
    auto t0 = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get local rank for GPU selection
    MPI_Comm local_comm;
    int local_rank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Select GPU based on local rank
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices < 1 || local_rank >= nDevices) {
        std::cerr << "Rank " << rank << " (local " << local_rank
                  << "): not enough GPUs (found " << nDevices << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int device = local_rank;
    cudaSetDevice(device);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);

    // Initialize logger
    Logger logger(output_dir, "mpi", collective_name);

    return MpiContext{size, rank, device, stream, init_time_ms, logger};
}

inline void finalize_mpi(MpiContext& ctx) {
    MPI_Barrier(MPI_COMM_WORLD);
    cudaStreamDestroy(ctx.stream);
    MPI_Finalize();
}
