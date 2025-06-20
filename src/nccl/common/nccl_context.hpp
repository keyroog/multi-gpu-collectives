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
    Logger logger;
};

inline NcclContext init_nccl(const std::string& output_dir, const std::string& collective_name) {
    // Initialize MPI
    MPI_Init(nullptr, nullptr);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Select GPU device based on rank
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices < size) {
        if (rank == 0) std::cerr << "Not enough GPUs for all ranks" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int device = rank % nDevices;
    cudaSetDevice(device);
    
    // Create NCCL communicator
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
    
    // Initialize logger
    Logger logger(output_dir, "nccl", collective_name);
    
    return NcclContext{size, rank, device, comm, stream, logger};
}
