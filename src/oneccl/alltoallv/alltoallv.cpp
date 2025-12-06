#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <numeric>

template <typename T>
void run_alltoallv(const std::vector<size_t>& send_counts,
                   const std::vector<size_t>& recv_counts,
                   const std::vector<size_t>& recv_displs,
                   size_t global_count,
                   int size,
                   int rank,
                   ccl::communicator& comm,
                   sycl::queue& q,
                   ccl::stream stream,
                   Logger& logger,
                   const std::string& data_type) {
    size_t total_send = std::accumulate(send_counts.begin(), send_counts.end(), 0UL);
    size_t total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0UL);

    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(total_send, q);
    auto recv_buf = sycl::malloc_device<T>(total_recv, q);
    
    // Copy send_counts to device memory for use in kernel
    auto d_send_counts = sycl::malloc_device<size_t>(size, q);
    q.memcpy(d_send_counts, send_counts.data(), size * sizeof(size_t)).wait();
    
    // initialize send buffer with rank and destination specific data
    auto e1 = q.submit([&](auto& h) {
        h.parallel_for(total_send, [=](auto global_id) {
            // Find which destination this element belongs to
            size_t cumulative = 0;
            int dest_rank = -1;
            size_t local_id = 0;
            
            for (int d = 0; d < size; ++d) {
                if (global_id < cumulative + d_send_counts[d]) {
                    dest_rank = d;
                    local_id = global_id - cumulative;
                    break;
                }
                cumulative += d_send_counts[d];
            }
            
            // Data pattern: sender*100000 + dest*10000 + local_id
            // This allows us to verify both sender and intended destination
            send_buf[global_id] = static_cast<T>(rank * 100000 + dest_rank * 10000 + local_id);
        });
    });
    
    // Initialize recv buffer to detect errors
    auto e2 = q.submit([&](auto& h) {
        h.parallel_for(total_recv, [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform alltoallv
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e1));
    deps.push_back(ccl::create_event(e2));
    auto attr = ccl::create_operation_attr<ccl::alltoallv_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::alltoallv(send_buf, send_counts, recv_buf, recv_counts, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << "Rank " << rank << " alltoallv time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms "
              << "(sent: " << total_send << ", received: " << total_recv << " elements)\n";
    
    // correctness check
    // Verify data from each source rank is correct and in the right position
    // First, copy recv_counts and recv_displs to device memory
    auto d_recv_counts = sycl::malloc_device<size_t>(size, q);
    auto d_recv_displs = sycl::malloc_device<size_t>(size, q);
    
    q.memcpy(d_recv_counts, recv_counts.data(), size * sizeof(size_t)).wait();
    q.memcpy(d_recv_displs, recv_displs.data(), size * sizeof(size_t)).wait();
    
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (int src_rank = 0; src_rank < size && passed; ++src_rank) {
                size_t src_count = d_recv_counts[src_rank];
                size_t src_offset = d_recv_displs[src_rank];
                
                for (size_t i = 0; i < src_count && passed; ++i) {
                    // Expected: src_rank sent (src_rank*100000 + rank*10000 + i) to this rank
                    T expected = static_cast<T>(src_rank * 100000 + rank * 10000 + i);
                    T actual = recv_buf[src_offset + i];
                    if (actual != expected) {
                        passed = false;
                    }
                }
            }
            acc[0] = passed ? static_cast<T>(1) : static_cast<T>(-1);
        });
    });
    q.wait_and_throw();
    
    // print result
    bool ok = false;
    {
        sycl::host_accessor acc(check_buf, sycl::read_only);
        ok = (acc[0] == static_cast<T>(1));
        if (ok) {
            std::cout << "Rank " << rank << " PASSED\n";
        } else {
            std::cout << "Rank " << rank << " FAILED\n";
        }
    }
    logger.log_result(data_type, global_count, size, rank, ok, elapsed_ms);
    
    // Print simple summary (only from rank 0 to avoid spam)
    if (rank == 0) {
        std::cout << "\nRank 0 send counts per destination:\n  ";
        for (int j = 0; j < size; ++j) {
            std::cout << send_counts[j] << (j + 1 == size ? "" : ", ");
        }
        std::cout << "\nRank 0 receive counts per source:\n  ";
        for (int s = 0; s < size; ++s) {
            std::cout << recv_counts[s] << (s + 1 == size ? "" : ", ");
        }
        std::cout << "\n";
    }
    
    // Free device memory for counts and displacements
    sycl::free(d_send_counts, q);
    sycl::free(d_recv_counts, q);
    sycl::free(d_recv_displs, q);
    
    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count");
    parser.parse();

    std::string dtype = parser.get<std::string>("--dtype");
    size_t global_count = parser.get<size_t>("--count");

    std::string output_dir;
    try {
        output_dir = parser.get<std::string>("--output");
    } catch (const std::runtime_error&) {
        output_dir = "";
    }
    
    // default value for global_count (still keep it small due to O(n²) scaling)
    if (global_count == 0) {
        global_count = 10000; // totale globale di elementi scambiati
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "alltoallv");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
    
    size_t per_rank_base = global_count / size;
    size_t per_rank_remainder = global_count % size;

    if (per_rank_base == 0) {
        if (rank == 0) {
            std::cerr << "Global count too small for size=" << size << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    size_t rank_total = per_rank_base + (static_cast<size_t>(rank) < per_rank_remainder ? 1 : 0);
    std::vector<size_t> send_counts(size);
    for (int dest = 0; dest < size; ++dest) {
        size_t send_base = rank_total / size;
        size_t send_remainder = rank_total % size;
        // spread the remainder across destinations based on sender rank
        size_t extra = ((dest + rank) % size) < send_remainder ? 1 : 0;
        send_counts[dest] = send_base + extra;
    }

    std::vector<size_t> recv_counts(size);
    std::vector<size_t> recv_displs(size);
    for (int src = 0; src < size; ++src) {
        size_t src_total = per_rank_base + (static_cast<size_t>(src) < per_rank_remainder ? 1 : 0);
        size_t src_base = src_total / size;
        size_t src_rem = src_total % size;
        size_t extra = ((rank + src) % size) < src_rem ? 1 : 0;
        size_t from_src = src_base + extra;
        recv_counts[src] = from_src;
        recv_displs[src] = (src == 0 ? 0 : recv_displs[src - 1] + recv_counts[src - 1]);
    }

    size_t effective_global_count = per_rank_base * size + per_rank_remainder;
    if (rank == 0 && per_rank_remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") not divisible by size (" << size
                  << "). Distributing remainder across ranks, effective_total="
                  << effective_global_count << ".\n";
    }

    if (rank == 0) {
        std::cout << "AllToAllV with " << size << " processes\n";
        std::cout << "Global elements: " << effective_global_count
                  << " (~" << rank_total << " per rank, distributed across peers)\n";
        std::cout << "Warning: This is the most memory-intensive collective operation!\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_alltoallv<int>(send_counts, recv_counts, recv_displs, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_alltoallv<float>(send_counts, recv_counts, recv_displs, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_alltoallv<double>(send_counts, recv_counts, recv_displs, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    return 0;
}
