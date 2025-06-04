#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>
#include <vector>

// Template wrapper for different data types
template <typename T>
void run_reduce_scatter(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                        Logger& logger, const std::string& data_type) {
    
    // ReduceScatter: each rank contributes count*size elements, reduces across ranks,
    // and each rank gets count elements of the reduced result (different segments)
    size_t total_elements = count * size;
    
    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(total_elements, q);
    auto recv_buf = sycl::malloc_device<T>(count, q);
    
    // initialize send buffer
    // Each rank contributes unique data for reduction
    auto e = q.submit([&](auto& h) {
        h.parallel_for(total_elements, [=](auto id) {
            // Data pattern: rank contribution for element id is (rank + 1) * (id + 1)
            send_buf[id] = static_cast<T>((rank + 1) * (id + 1));
        });
        
        // Initialize recv buffer to detect errors
        h.parallel_for(count, [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform reduce_scatter with sum reduction
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::reduce_scatter(send_buf, recv_buf, count, ccl::reduction::sum, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " reduce_scatter time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms "
              << "(received: " << count << " elements of reduced data)\n";
    
    // correctness check
    // Each rank receives elements [rank*count : (rank+1)*count) of the reduced array
    // Expected value for element i in rank r's segment: sum over all ranks of (rank+1)*(segment_start+i+1)
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            
            for (size_t i = 0; i < count && passed; ++i) {
                // Global index for this element in the conceptual total array
                size_t global_idx = rank * count + i;
                
                // Calculate expected sum: sum over all ranks of (rank+1)*(global_idx+1)
                T expected = static_cast<T>(0);
                for (int r = 0; r < size; ++r) {
                    expected += static_cast<T>((r + 1) * (global_idx + 1));
                }
                
                T actual = recv_buf[i];
                if (actual != expected) {
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
    
    // Print segment information (only from rank 0 to avoid spam)
    if (rank == 0) {
        std::cout << "\nReduceScatter operation details:\n";
        std::cout << "  Each rank contributes: " << total_elements << " elements\n";
        std::cout << "  Each rank receives: " << count << " elements (different segments)\n";
        std::cout << "  Total computation: " << size << " ranks × " << total_elements << " = " 
                  << (size * total_elements) << " reduction operations\n";
        
        std::cout << "\nSegment distribution:\n";
        for (int r = 0; r < size; ++r) {
            std::cout << "  Rank " << r << " receives elements [" << (r * count) 
                      << " : " << ((r + 1) * count - 1) << "] of reduced array\n";
        }
    }
    
    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count");
    parser.parse();

    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");

    std::string output_dir;
    try {
        output_dir = parser.get<std::string>("--output");
    } catch (const std::runtime_error&) {
        output_dir = "";
    }
    
    // default value for count
    if (count == 0) {
        count = 1024 * 1024; // 1M elements per rank (total 1M*size per rank contribution)
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "reduce_scatter");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
    
    if (rank == 0) {
        std::cout << "ReduceScatter with " << size << " processes\n";
        std::cout << "Count per rank: " << count << " elements\n";
        std::cout << "Total elements per rank contribution: " << (count * size) << "\n";
        std::cout << "Reduction operation: sum\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_reduce_scatter<int>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_reduce_scatter<float>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_reduce_scatter<double>(count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
