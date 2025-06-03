#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

// Template wrapper for different data types
template <typename T>
void run_allgather(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                   Logger& logger, const std::string& data_type) {
    // allocate device buffers
    // send_buf contains count elements from this rank
    // recv_buf will contain count*size elements (count from each rank)
    auto send_buf = sycl::malloc_device<T>(count, q);
    auto recv_buf = sycl::malloc_device<T>(count * size, q);
    
    // initialize send buffer with rank-specific data
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = static_cast<T>(rank * 100 + id);  // unique data per rank
            // Initialize recv_buf to detect errors
            for (int r = 0; r < size; ++r) {
                recv_buf[r * count + id] = static_cast<T>(-1);
            }
        });
    });
    
    // perform allgather
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::allgather_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::allgather(send_buf, recv_buf, count, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " allgather time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check
    // Each segment of recv_buf should contain data from the corresponding rank
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (int r = 0; r < size && passed; ++r) {
                for (size_t i = 0; i < count && passed; ++i) {
                    T expected = static_cast<T>(r * 100 + i);
                    if (recv_buf[r * count + i] != expected) {
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
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--output");
    parser.parse();
    
    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");
    std::string output_dir = parser.get<std::string>("--output");
    
    // default value for count
    if (count == 0) {
        count = 10 * 1024 * 1024; // Default value if not provided
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "allgather");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_allgather<int>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_allgather<float>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_allgather<double>(count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
