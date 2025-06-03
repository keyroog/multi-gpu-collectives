#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

// Template wrapper for different data types
template <typename T>
void run_broadcast(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                   Logger& logger, const std::string& data_type, int root_rank) {
    // allocate device buffer
    // For broadcast, we use the same buffer for send and receive
    auto buf = sycl::malloc_device<T>(count, q);
    
    // initialize buffer
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            if (rank == root_rank) {
                // Root rank initializes with meaningful data
                buf[id] = static_cast<T>(root_rank * 1000 + id);
            } else {
                // Non-root ranks initialize with sentinel values
                buf[id] = static_cast<T>(-1);
            }
        });
    });
    
    // perform broadcast
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::broadcast(buf, count, root_rank, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " broadcast time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check
    // All ranks should now have the same data that root_rank originally had
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (size_t i = 0; i < count && passed; ++i) {
                T expected = static_cast<T>(root_rank * 1000 + i);
                if (buf[i] != expected) {
                    passed = false;
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
    
    sycl::free(buf, q);
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--output");
    parser.parse();
    
    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");
    std::string output_dir = parser.get<std::string>("--output");
    
    // Handle root rank with default value
    int root_rank = 0; // Default to rank 0
    try {
        root_rank = parser.get<int>("--root");
    } catch (const std::runtime_error&) {
        // Use default value if --root not provided
        root_rank = 0;
    }
    
    // default values
    if (count == 0) {
        count = 10 * 1024 * 1024; // Default value if not provided
    }
    
    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "broadcast");
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
        std::cout << "Broadcasting from rank " << root_rank << " to " << size << " processes\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_broadcast<int>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "float") {
        run_broadcast<float>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "double") {
        run_broadcast<double>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
