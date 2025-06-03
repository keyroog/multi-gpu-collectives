#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

// Template wrapper for different data types
template <typename T>
void run_scatter(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                 Logger& logger, const std::string& data_type, int root_rank) {
    // allocate device buffers
    // send_buf is only meaningful on root rank - contains count*size elements
    // recv_buf is meaningful on all ranks - contains count elements
    auto send_buf = sycl::malloc_device<T>(count * size, q);
    auto recv_buf = sycl::malloc_device<T>(count, q);
    
    // initialize buffers
    auto e = q.submit([&](auto& h) {
        h.parallel_for(sycl::range<1>(count * size), [=](auto global_id) {
            if (rank == root_rank) {
                // Root rank prepares data for all ranks
                int dest_rank = global_id / count;  // which rank this data goes to
                int local_id = global_id % count;   // position within the segment
                
                // Data pattern: root sends (root*1000 + dest*100 + local_id) to dest_rank
                send_buf[global_id] = static_cast<T>(root_rank * 1000 + dest_rank * 100 + local_id);
            } else {
                // Non-root ranks don't use send_buf, but initialize it anyway
                send_buf[global_id] = static_cast<T>(-1);
            }
        });
        
        // Initialize receive buffer for all ranks
        h.parallel_for(sycl::range<1>(count), [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform scatter
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::scatter(send_buf, recv_buf, count, root_rank, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " scatter time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check - all ranks verify their received data
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (size_t i = 0; i < count && passed; ++i) {
                // Expected: root sent (root*1000 + rank*100 + i) to this rank
                T expected = static_cast<T>(root_rank * 1000 + rank * 100 + i);
                if (recv_buf[i] != expected) {
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
    auto ctx = init_oneccl(output_dir, "scatter");
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
        std::cout << "Scattering from rank " << root_rank << " to " << size << " processes\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_scatter<int>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "float") {
        run_scatter<float>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "double") {
        run_scatter<double>(count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
