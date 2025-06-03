#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

// Template wrapper for different data types
template <typename T>
void run_reduce(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                Logger& logger, const std::string& data_type, int root_rank) {
    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(count, q);
    auto recv_buf = sycl::malloc_device<T>(count, q);
    
    // initialize send buffer - each rank contributes different data
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            // Each rank contributes (rank + 1) * (id + 1) to make verification easier
            send_buf[id] = static_cast<T>((rank + 1) * (id + 1));
            // Initialize recv buffer
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // compute expected result (only needed for root rank verification)
    T check_factor = static_cast<T>(0);
    for (int r = 0; r < size; ++r) {
        check_factor += static_cast<T>(r + 1);
    }
    
    // perform reduce (sum operation)
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::reduce_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::reduce(send_buf, recv_buf, count, ccl::reduction::sum, root_rank, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " reduce time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check - only root rank has valid results
    if (rank == root_rank) {
        sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
        q.submit([&](auto& h) {
            sycl::accessor acc(check_buf, h, sycl::write_only);
            h.single_task([=]() {
                bool passed = true;
                for (size_t i = 0; i < count && passed; ++i) {
                    // Expected: sum of (r+1)*(i+1) for r from 0 to size-1
                    // = (i+1) * sum(r+1) = (i+1) * check_factor
                    T expected = static_cast<T>((i + 1)) * check_factor;
                    if (recv_buf[i] != expected) {
                        passed = false;
                    }
                }
                acc[0] = passed ? static_cast<T>(1) : static_cast<T>(-1);
            });
        });
        q.wait_and_throw();
        
        // print result (only root rank)
        {
            sycl::host_accessor acc(check_buf, sycl::read_only);
            if (acc[0] == static_cast<T>(1)) {
                std::cout << "Root rank " << root_rank << " PASSED\n";
            } else {
                std::cout << "Root rank " << root_rank << " FAILED\n";
            }
        }
    } else {
        // Non-root ranks should not have valid data in recv_buf after reduce
        std::cout << "Rank " << rank << " (non-root) completed reduce operation\n";
    }
    
    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--output").add<int>("--root");
    parser.parse();
    
    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");
    std::string output_dir = parser.get<std::string>("--output");
    int root_rank = parser.get<int>("--root");
    
    // default values
    if (count == 0) {
        count = 10 * 1024 * 1024; // Default value if not provided
    }
    
    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "reduce");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
    
    // Validate root rank
    if (root_rank < 0 || root_rank >= size) {
        if (rank == 0) {
            std::cerr << "Invalid root rank " << root_rank << ". Must be between 0 and " << (size-1) << std::endl;
        }
        exit(-1);
    }
    
    if (rank == 0) {
        std::cout << "Reducing to rank " << root_rank << " from " << size << " processes\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_reduce<int>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "float") {
        run_reduce<float>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "double") {
        run_reduce<double>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
