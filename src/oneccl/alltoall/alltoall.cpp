#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

// Template wrapper for different data types
template <typename T>
void run_alltoall(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                  Logger& logger, const std::string& data_type) {
    // allocate device buffers
    // send_buf contains count*size elements (count elements for each destination rank)
    // recv_buf will contain count*size elements (count elements from each source rank)
    auto send_buf = sycl::malloc_device<T>(count * size, q);
    auto recv_buf = sycl::malloc_device<T>(count * size, q);
    
    // initialize send buffer - each segment goes to a different rank
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count * size, [=](auto global_id) {
            int dest_rank = global_id / count;  // which rank this element goes to
            int local_id = global_id % count;   // position within the segment
            
            // Data pattern: rank sends (rank*1000 + dest_rank*100 + local_id) to dest_rank
            send_buf[global_id] = static_cast<T>(rank * 1000 + dest_rank * 100 + local_id);
            
            // Initialize recv_buf to detect errors
            recv_buf[global_id] = static_cast<T>(-1);
        });
    });
    
    // perform alltoall
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::alltoall_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::alltoall(send_buf, recv_buf, count, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " alltoall time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check
    // After alltoall, recv_buf[src_rank*count + i] should contain data that src_rank sent to this rank
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (int src_rank = 0; src_rank < size && passed; ++src_rank) {
                for (size_t i = 0; i < count && passed; ++i) {
                    // Expected value: src_rank sent (src_rank*1000 + rank*100 + i) to this rank
                    T expected = static_cast<T>(src_rank * 1000 + rank * 100 + i);
                    T actual = recv_buf[src_rank * count + i];
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
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED\n";
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
        count = 1024 * 1024; // Smaller default for alltoall due to O(n²) communication
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "alltoall");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_alltoall<int>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_alltoall<float>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_alltoall<double>(count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
