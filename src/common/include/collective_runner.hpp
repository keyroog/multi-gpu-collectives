#pragma once
#include <iostream>
#include <mpi.h>
#include "common/include/arg_parser.hpp"
#include "oneapi/ccl.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <vector>
#include <string>
#include "common/include/logger.hpp"

namespace collective_runner {

inline void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) MPI_Finalize();
}

// Template to run any collective operation
template <typename T>
void run_collective(const std::string& coll_name,
                    const std::string& dtype,
                    size_t count,
                    int size,
                    int rank,
                    ccl::communicator& comm,
                    sycl::queue& q,
                    const std::string& logpath,
                    const std::function<void(T*, T*, size_t, ccl::communicator&, sycl::queue&, const std::vector<ccl::event>&)>& op) {
    // allocate device buffers
    T* send_buf = sycl::malloc_device<T>(count, q);
    T* recv_buf = sycl::malloc_device<T>(count, q);
    // initialize buffers
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = static_cast<T>(rank + id + 1);
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    // compute expected sum
    T check_sum = static_cast<T>(0);
    for (int i = 1; i <= size; ++i) check_sum += static_cast<T>(i);
    // perform collective
    std::vector<ccl::event> deps{ccl::create_event(e)};
    auto t_start = std::chrono::high_resolution_clock::now();
    op(send_buf, recv_buf, count, comm, q, deps);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "Rank " << rank << " time: " << elapsed_ms << " ms\n";
    // correctness check
    sycl::buffer<T> check_buf(count);
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            if (recv_buf[id] != static_cast<T>(check_sum + size * id)) acc[id] = static_cast<T>(-1);
        });
    });
    q.wait_and_throw();
    // print result
    {
        sycl::host_accessor acc(check_buf, sycl::read_only);
        size_t i = 0;
        for (; i < count; ++i) {
            if (acc[i] == static_cast<T>(-1)) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == count) std::cout << "PASSED\n";
    }
    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);
    // log results
    bool passed = (i == count);
    Logger::append(logpath, {coll_name, dtype, std::to_string(count), std::to_string(size), std::to_string(rank), std::to_string(elapsed_ms), passed ? "PASSED" : "FAILED"});
}

// Main template that sets up environment and dispatches based on dtype
inline int main_collective(int argc, char* argv,
                           const std::function<void(const std::string&, const std::string&, size_t, const std::string&, int, int, ccl::communicator&, sycl::queue&)> &dispatch) {
    // parse common args
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--log");
    parser.parse();
    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");
    std::string logpath = parser.get<std::string>("--log");
    // initialize CCL and MPI
    ccl::init();
    MPI_Init(nullptr, nullptr);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    atexit(mpi_finalize);
    // setup SYCL devices and queue
    std::vector<sycl::device> devices;
    for (auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos) {
            for (auto &d : platform.get_devices())
                if (d.is_gpu()) devices.push_back(d);
        }
    }
    if (devices.size() < static_cast<size_t>(size)) {
        std::cerr << "Not enough devices for all ranks" << std::endl;
        return -1;
    }
    sycl::context context(devices);
    sycl::queue queue(context, devices[rank], {sycl::property::queue::in_order()});
    // create CCL communicator
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        addr = kvs->get_address();
        MPI_Bcast((void*)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast((void*)addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(addr);
    }
    auto dev = ccl::create_device(queue.get_device());
    auto ctx = ccl::create_context(queue.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    // dispatch to specific collective implementation
    dispatch(dtype, std::to_string(count), count, logpath, size, rank, comm, queue);
    MPI_Finalize();
    return 0;
}

} // namespace collective_runner
