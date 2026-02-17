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
    double init_time_ms;
    Logger logger;
};

sycl::device pick_device_for_rank(int rank, const std::string& gpu_mode) {
    std::vector<sycl::device> candidates;

    for (const auto& plat : sycl::platform::get_platforms()) {
        auto name = plat.get_info<sycl::info::platform::name>();

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

    // Create a node-local communicator to get the local rank
    MPI_Comm local_comm;
    int local_rank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    // Use local rank to select the device on this node
    sycl::device dev = pick_device_for_rank(local_rank, gpu_mode);

    MPI_Comm_free(&local_comm);

    // Context e queue SOLO su quel (sub)device (best practice)
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, { sycl::property::queue::in_order() });

    // KVS + communicator + stream
    double t_init_start = MPI_Wtime();
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank == 0) { kvs = ccl::create_main_kvs(); addr = kvs->get_address(); }
    MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(addr);

    auto ccl_dev = ccl::create_device(q.get_device());  // subdevice se sei in modalità tile
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm    = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto stream  = ccl::create_stream(q);
    double t_init_end = MPI_Wtime();
    double init_time_ms = (t_init_end - t_init_start) * 1000.0;

    Logger logger(output_dir, "oneccl", collective_name);
    return OneCCLContext{size, rank, std::move(q), std::move(comm), std::move(stream), init_time_ms, std::move(logger)};
}

inline void finalize_oneccl(OneCCLContext& ctx) {
    // Sincronizza tutti i rank prima del cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // OneCCL cleanup: stream, comm distrutti automaticamente (RAII)
    // Ma dobbiamo fare MPI_Finalize
    MPI_Finalize();
}