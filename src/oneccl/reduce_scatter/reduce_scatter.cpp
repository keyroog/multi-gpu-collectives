#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>
#include <vector>

template <typename T>
void run_reduce_scatter(size_t local_count, size_t global_count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                        Logger& logger, const std::string& data_type) {
    
    // ReduceScatter: each rank contributes count*size elements, reduces across ranks,
    // and each rank gets count elements of the reduced result (different segments)
    size_t total_elements = local_count * size;
    
    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(total_elements, q);
    auto recv_buf = sycl::malloc_device<T>(local_count, q);
    
    // initialize send buffer
    // Each rank contributes unique data for reduction
    auto e = q.submit([&](auto& h) {
        h.parallel_for(total_elements, [=](auto id) {
            // Data pattern: rank contribution for element id is (rank + 1) * (id + 1)
            send_buf[id] = static_cast<T>((rank + 1) * (id + 1));
        });
        
        // Initialize recv buffer to detect errors
        h.parallel_for(local_count, [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform reduce_scatter with sum reduction
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::reduce_scatter(send_buf, recv_buf, local_count, ccl::reduction::sum, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << "Rank " << rank << " reduce_scatter time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms "
              << "(received: " << local_count << " elements of reduced data)\n";
    
    // correctness check
    // Each rank receives elements [rank*count : (rank+1)*count) of the reduced array
    // Expected value for element i in rank r's segment: sum over all ranks of (rank+1)*(segment_start+i+1)
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            
            for (size_t i = 0; i < local_count && passed; ++i) {
                // Global index for this element in the conceptual total array
                size_t global_idx = rank * local_count + i;
                
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
    
    // Print segment information (only from rank 0 to avoid spam)
    if (rank == 0) {
        std::cout << "\nReduceScatter operation details:\n";
        std::cout << "  Each rank contributes: " << total_elements << " elements\n";
        std::cout << "  Each rank receives: " << local_count << " elements (different segments)\n";
        std::cout << "  Total computation: " << size << " ranks × " << total_elements << " = " 
                  << (size * total_elements) << " reduction operations\n";
        
        std::cout << "\nSegment distribution:\n";
        for (int r = 0; r < size; ++r) {
            std::cout << "  Rank " << r << " receives elements [" << (r * local_count) 
                      << " : " << ((r + 1) * local_count - 1) << "] of reduced array\n";
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
    size_t global_count = parser.get<size_t>("--count");

    std::string output_dir;
    try {
        output_dir = parser.get<std::string>("--output");
    } catch (const std::runtime_error&) {
        output_dir = "";
    }
    
    // default value for count
    if (global_count == 0) {
        global_count = 1024 * 1024; // totale globale di elementi ridotti
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "reduce_scatter");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;

    size_t local_count = global_count / size;
    size_t remainder   = global_count % size;
    size_t effective_global_count = local_count * size;

    if (local_count == 0) {
        if (rank == 0) {
            std::cerr << "Global count too small for size=" << size << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0 && remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") is not divisible by size (" << size
                  << "). Using local_count=" << local_count
                  << " and ignoring last " << remainder << " elements.\n";
    }
    
    if (rank == 0) {
        std::cout << "ReduceScatter with " << size << " processes\n";
        std::cout << "Global elements: " << effective_global_count << " (" << local_count << " per rank)\n";
        std::cout << "Total elements per rank contribution: " << (local_count * size) << "\n";
        std::cout << "Reduction operation: sum\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_reduce_scatter<int>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_reduce_scatter<float>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_reduce_scatter<double>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    return 0;
}
