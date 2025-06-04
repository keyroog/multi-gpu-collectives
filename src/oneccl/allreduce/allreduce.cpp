#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"  // retain for run_allreduce
#include <chrono>                    // <<< aggiunto
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>

// Template wrapper for different data types
template <typename T>
void run_allreduce(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                  Logger& logger, const std::string& data_type) {
    // allocate device buffers
    auto send_buf = sycl::malloc_device<T>(count, q);
    auto recv_buf = sycl::malloc_device<T>(count, q);
    // initialize buffers
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = static_cast<T>(rank + id + 1);
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    // compute expected sum
    T check_sum = static_cast<T>(0);
    for (int i = 1; i <= size; ++i) check_sum += static_cast<T>(i);
    // perform allreduce
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::allreduce(send_buf, recv_buf, count, ccl::reduction::sum, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " allreduce time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    // correctness check
    sycl::buffer<T> check_buf(count);
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            if (recv_buf[id] != static_cast<T>(check_sum + size * id)) acc[id] = static_cast<T>(-1);
        });
    });
    q.wait_and_throw();
    // print result
    {
        sycl::host_accessor acc(check_buf, sycl::read_only);
        size_t i = 0;
        for (; i < count; ++i) if (acc[i] == static_cast<T>(-1)) { std::cout << "FAILED\n"; break; }
        if (i == count) std::cout << "PASSED\n";
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
        output_dir = ""; // logging disabled if not provided
    }
    
    //default value for count
    if (count == 0) {
        count = 10 * 1024 * 1024; // Default value if not provided
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "allreduce");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_allreduce<int>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_allreduce<float>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_allreduce<double>(count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    return 0;
}