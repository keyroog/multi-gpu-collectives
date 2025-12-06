#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

template <typename T>
void run_alltoall(size_t count_per_dest, size_t global_count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                  Logger& logger, const std::string& data_type) {
    // allocate device buffers
    // send_buf contains count*size elements (count elements for each destination rank)
    // recv_buf will contain count*size elements (count elements from each source rank)
    auto send_buf = sycl::malloc_device<T>(count_per_dest * size, q);
    auto recv_buf = sycl::malloc_device<T>(count_per_dest * size, q);
    
    // initialize send buffer - each segment goes to a different rank
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count_per_dest * size, [=](auto global_id) {
            int dest_rank = global_id / count_per_dest;  // which rank this element goes to
            int local_id = global_id % count_per_dest;   // position within the segment
            
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
    ccl::alltoall(send_buf, recv_buf, count_per_dest, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    std::cout << "Rank " << rank << " alltoall time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check
    // After alltoall, recv_buf[src_rank*count + i] should contain data that src_rank sent to this rank
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (int src_rank = 0; src_rank < size && passed; ++src_rank) {
                for (size_t i = 0; i < count_per_dest && passed; ++i) {
                    // Expected value: src_rank sent (src_rank*1000 + rank*100 + i) to this rank
                    T expected = static_cast<T>(src_rank * 1000 + rank * 100 + i);
                    T actual = recv_buf[src_rank * count_per_dest + i];
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
    bool ok = false;
    {
        sycl::host_accessor acc(check_buf, sycl::read_only);
        ok = (acc[0] == static_cast<T>(1));
        if (ok) {
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED\n";
        }
    }
    logger.log_result(data_type, global_count, size, rank, ok, elapsed_ms);
    
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
        global_count = 1024 * 1024; // totale globale di elementi scambiati
    }

    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "alltoall");
    int size = ctx.size;
    int rank = ctx.rank;
    auto& q = ctx.q;
    auto& comm = ctx.comm;
    auto& stream = ctx.stream;
    auto& logger = ctx.logger;

    size_t denom = static_cast<size_t>(size) * static_cast<size_t>(size);
    size_t count_per_dest = global_count / denom;
    size_t remainder      = global_count % denom;
    size_t effective_global_count = count_per_dest * denom;

    if (count_per_dest == 0) {
        if (rank == 0) {
            std::cerr << "Global count too small for size=" << size << " (needs at least size^2 elements).\n";
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0 && remainder != 0) {
        std::cerr << "Warning: global_count (" << global_count
                  << ") is not divisible by size^2 (" << denom
                  << "). Using count_per_dest=" << count_per_dest
                  << " and ignoring last " << remainder << " elements.\n";
    }

    if (rank == 0) {
        std::cout << "Alltoall with " << size << " processes\n";
        std::cout << "Global elements: " << effective_global_count
                  << " (" << count_per_dest << " per destination per rank)\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_alltoall<int>(count_per_dest, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        run_alltoall<float>(count_per_dest, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        run_alltoall<double>(count_per_dest, effective_global_count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    return 0;
}
