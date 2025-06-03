#pragma once

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "../../common/include/logger.hpp"

struct OneCCLContext {
    int size;
    int rank;
    sycl::queue q;
    ccl::communicator comm;
    ccl::stream stream;
    Logger logger;
};

inline OneCCLContext init_oneccl(const std::string& output_dir, const std::string& collective_name) {
    // Initialize CCL and MPI
    ccl::init();
    MPI_Init(nullptr, nullptr);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    atexit([](){
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (!is_finalized) MPI_Finalize();
    });

    // Select Level-Zero GPU devices
    std::vector<sycl::device> devices;
    auto platforms = sycl::platform::get_platforms();
    for (const auto& platform : platforms) {
        auto pname = platform.get_info<sycl::info::platform::name>();
        if (pname.find("Level-Zero") != std::string::npos) {
            for (const auto& device : platform.get_devices()) {
                if (device.is_gpu()) devices.push_back(device);
            }
        }
    }
    if (devices.size() < static_cast<size_t>(size)) {
        if (rank == 0) std::cerr << "Not enough devices for all ranks" << std::endl;
        std::exit(-1);
    }

    // Create SYCL queue for this rank's device
    sycl::context context(devices);
    sycl::queue q(context, devices[rank], {sycl::property::queue::in_order()});

    // Setup CCL KVS and communicator
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    auto stream = ccl::create_stream(q);

    // Initialize logger
    Logger logger(output_dir, "oneccl", collective_name);

    return OneCCLContext{size, rank, q, comm, stream, logger};
}
