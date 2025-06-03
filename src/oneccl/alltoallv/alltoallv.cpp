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

// Template wrapper for different data types
template <typename T>
void run_alltoallv(size_t base_count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                   Logger& logger, const std::string& data_type) {
    
    // Each rank sends different amounts to different destinations
    // Pattern: rank r sends (base_count + r*100 + dest*50) elements to rank dest
    std::vector<size_t> send_counts(size);
    std::vector<size_t> send_displs(size);
    std::vector<size_t> recv_counts(size);
    std::vector<size_t> recv_displs(size);
    
    // Calculate send counts and displacements
    size_t total_send = 0;
    for (int dest = 0; dest < size; ++dest) {
        send_counts[dest] = base_count + rank * 100 + dest * 50;
        send_displs[dest] = total_send;
        total_send += send_counts[dest];
    }
    
    // Calculate receive counts and displacements
    // We need to know what each source rank will send to us
    size_t total_recv = 0;
    for (int src = 0; src < size; ++src) {
        recv_counts[src] = base_count + src * 100 + rank * 50;
        recv_displs[src] = total_recv;
        total_recv += recv_counts[src];
    }
    
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
    ccl::alltoallv(send_buf, send_counts, send_displs, recv_buf, recv_counts, recv_displs, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati - use total elements for meaningful comparison
    logger.log_result(data_type, total_recv, size, rank, elapsed_ms);
    
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
    {
        sycl::host_accessor acc(check_buf, sycl::read_only);
        if (acc[0] == static_cast<T>(1)) {
            std::cout << "Rank " << rank << " PASSED\n";
        } else {
            std::cout << "Rank " << rank << " FAILED\n";
        }
    }
    
    // Print detailed communication matrix (only from rank 0 to avoid spam)
    if (rank == 0) {
        std::cout << "\nCommunication Matrix (elements sent from rank i to rank j):\n";
        std::cout << "     ";
        for (int j = 0; j < size; ++j) {
            std::cout << std::setw(8) << ("To_" + std::to_string(j));
        }
        std::cout << "\n";
        
        for (int i = 0; i < size; ++i) {
            std::cout << "From_" << i << ":";
            for (int j = 0; j < size; ++j) {
                size_t elements = base_count + i * 100 + j * 50;
                std::cout << std::setw(8) << elements;
            }
            std::cout << "\n";
        }
        
        std::cout << "\nTotal elements per rank:\n";
        for (int r = 0; r < size; ++r) {
            size_t rank_total_send = 0;
            size_t rank_total_recv = 0;
            for (int other = 0; other < size; ++other) {
                rank_total_send += base_count + r * 100 + other * 50;  // what rank r sends
                rank_total_recv += base_count + other * 100 + r * 50;  // what rank r receives
            }
            std::cout << "  Rank " << r << ": sends " << rank_total_send 
                      << ", receives " << rank_total_recv << " elements\n";
        }
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
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--output");
    parser.parse();
    
    std::string dtype = parser.get<std::string>("--dtype");
    size_t base_count = parser.get<size_t>("--count");
    std::string output_dir = parser.get<std::string>("--output");
    
    // default value for base_count (smaller due to O(n²) scaling with variable sizes)
    if (base_count == 0) {
        base_count = 10000; // 10K base elements, much smaller due to complexity
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "alltoallv");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
    
    if (rank == 0) {
        std::cout << "AllToAllV with " << size << " processes\n";
        std::cout << "Base count: " << base_count << " elements\n";
        std::cout << "Each rank sends (base + sender*100 + dest*50) elements to each destination\n";
        std::cout << "Warning: This is the most memory-intensive collective operation!\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_alltoallv<int>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_alltoallv<float>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_alltoallv<double>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
