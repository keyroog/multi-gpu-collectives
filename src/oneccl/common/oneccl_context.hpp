#pragma once

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
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

sycl::device pick_device_for_rank(int rank, const std::string& gpu_mode) {
    std::vector<sycl::device> candidates;

    for (const auto& plat : sycl::platform::get_platforms()) {
        auto name = plat.get_info<sycl::info::platform::name>();
        if (name.find("Level-Zero") == std::string::npos) continue;

        for (const auto& root : plat.get_devices()) {
            if (!root.is_gpu()) continue;

            if (gpu_mode == "gpu") {
                candidates.push_back(root);
            }
            else if (gpu_mode == "tile") {
                // ogni subdevice corrisponde a un tile/stack
                auto tiles = root.create_sub_devices<
                    sycl::info::partition_property::partition_by_affinity_domain>(
                    sycl::info::partition_affinity_domain::next_partitionable);
                candidates.insert(candidates.end(), tiles.begin(), tiles.end());
            }
        }
    }

    if (rank >= (int)candidates.size()) {
        throw std::runtime_error("Not enough (sub)devices for ranks");
    }
    return candidates[rank];
}


inline OneCCLContext init_oneccl(const std::string& output_dir,
                                 const std::string& collective_name,
                                 const std::string& gpu_mode = "gpu") {
    ccl::init();
    MPI_Init(nullptr, nullptr);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Scegli il (sub)device per questo rank
    sycl::device dev = pick_device_for_rank(rank, gpu_mode);

    // Context e queue SOLO su quel (sub)device (best practice)
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, { sycl::property::queue::in_order() });

    // KVS + communicator + stream
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank == 0) { kvs = ccl::create_main_kvs(); addr = kvs->get_address(); }
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(addr);

    auto ccl_dev = ccl::create_device(q.get_device());  // subdevice se sei in modalità tile
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm    = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto stream  = ccl::create_stream(q);

    Logger logger(output_dir, "oneccl", collective_name);
    return OneCCLContext{size, rank, std::move(q), std::move(comm), std::move(stream), std::move(logger)};
}