#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>
#include <vector>
#include <numeric>

// Template wrapper for different data types
template <typename T>
void run_allgatherv(size_t base_count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                    Logger& logger, const std::string& data_type) {
    // Each rank contributes a different amount of data
    // Rank r contributes base_count + r*1000 elements
    size_t send_count = base_count + rank * 1000;
    
    // Calculate receive counts and displacements
    std::vector<size_t> recv_counts(size);
    std::vector<size_t> recv_displs(size);
    
    for (int r = 0; r < size; ++r) {
        recv_counts[r] = base_count + r * 1000;
    }
    
    recv_displs[0] = 0;
    for (int r = 1; r < size; ++r) {
        recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
    }
    
    size_t total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0UL);
    
    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(send_count, q);
    auto recv_buf = sycl::malloc_device<T>(total_recv_count, q);
    
    // initialize send buffer with rank-specific data
    auto e = q.submit([&](auto& h) {
        h.parallel_for(send_count, [=](auto id) {
            // Each rank sends unique data: rank*10000 + id
            send_buf[id] = static_cast<T>(rank * 10000 + id);
        });
        
        // Initialize recv buffer to detect errors
        h.parallel_for(total_recv_count, [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform allgatherv
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::allgatherv(send_buf, send_count, recv_buf, recv_counts, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati - use total elements for meaningful comparison
    logger.log_result(data_type, total_recv_count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " allgatherv time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms "
              << "(sent: " << send_count << ", received: " << total_recv_count << " elements)\n";
    
    // correctness check
    // Verify data from each source rank is in the correct position
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
                    T expected = static_cast<T>(src_rank * 10000 + i);
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
    
    // Print detailed info about data distribution
    if (rank == 0) {
        std::cout << "\nData distribution:\n";
        for (int r = 0; r < size; ++r) {
            std::cout << "  Rank " << r << ": " << recv_counts[r] << " elements at offset " << recv_displs[r] << "\n";
        }
        std::cout << "  Total: " << total_recv_count << " elements\n";
    }
    
    // Free device memory for counts and displacements
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
    
    // default value for base_count
    if (base_count == 0) {
        base_count = 1000000; // 1M base elements, each rank adds more
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "allgatherv");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
    
    if (rank == 0) {
        std::cout << "AllGatherV with " << size << " processes\n";
        std::cout << "Base count: " << base_count << " elements per rank\n";
        std::cout << "Each rank contributes base_count + rank*1000 elements\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_allgatherv<int>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_allgatherv<float>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_allgatherv<double>(base_count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
