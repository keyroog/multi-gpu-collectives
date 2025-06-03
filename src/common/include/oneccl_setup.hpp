#pragma once

#include <vector>
#include <iostream>
#include <mpi.h>
#include "oneapi/ccl.hpp"

class OneCCLSetup {
private:
    std::vector<sycl::device> devices;
    std::vector<sycl::queue> queues;
    int size, rank;
    bool initialized;
    
public:
    struct SetupResult {
        sycl::queue& queue;
        ccl::communicator comm;
        ccl::stream stream;
        int size;
        int rank;
    };
    
    OneCCLSetup();
    ~OneCCLSetup();
    
    SetupResult initialize();
    
private:
    void setup_mpi();
    void setup_sycl_devices();
    void setup_ccl_communicator(sycl::queue& q, ccl::communicator& comm, ccl::stream& stream);
    static void mpi_finalize();
};
