#include "../common/oneccl_context.hpp"
#include "oneapi/ccl.hpp"
#include <chrono>
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>
#include <iomanip>

template <typename T>
void run_scatter(size_t local_count, size_t global_count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
                 Logger& logger, const std::string& data_type, int root_rank) {
    // allocate device buffers
    // send_buf is only meaningful on root rank - contains count*size elements
    // recv_buf is meaningful on all ranks - contains count elements
    auto send_buf = sycl::malloc_device<T>(local_count * size, q);
    auto recv_buf = sycl::malloc_device<T>(local_count, q);
    
    // initialize buffers
    auto e = q.submit([&](auto& h) {
        h.parallel_for(sycl::range<1>(local_count * size), [=](auto global_id) {
            if (rank == root_rank) {
                // Root rank prepares data for all ranks
                int dest_rank = global_id / local_count;  // which rank this data goes to
                int local_id = global_id % local_count;   // position within the segment
                
                // Data pattern: root sends (root*1000 + dest*100 + local_id) to dest_rank
                send_buf[global_id] = static_cast<T>(root_rank * 1000 + dest_rank * 100 + local_id);
            } else {
                // Non-root ranks don't use send_buf, but initialize it anyway
                send_buf[global_id] = static_cast<T>(-1);
            }
        });
        
        // Initialize receive buffer for all ranks
        h.parallel_for(sycl::range<1>(local_count), [=](auto id) {
            recv_buf[id] = static_cast<T>(-1);
        });
    });
    
    // perform scatter
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    ccl::scatter(send_buf, recv_buf, local_count, root_rank, comm, stream, attr, deps).wait();
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    
    // Log dei risultati
    logger.log_result(data_type, global_count, size, rank, elapsed_ms);
    
    std::cout << "Rank " << rank << " scatter time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    
    // correctness check - all ranks verify their received data
    sycl::buffer<T> check_buf(1);  // Single flag for pass/fail
    q.submit([&](auto& h) {
        sycl::accessor acc(check_buf, h, sycl::write_only);
        h.single_task([=]() {
            bool passed = true;
            for (size_t i = 0; i < local_count && passed; ++i) {
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
    
    // Handle root rank with default value
    int root_rank = 0; // Default to rank 0
    try {
        root_rank = parser.get<int>("--root");
    } catch (const std::runtime_error&) {
        // Use default value if --root not provided
        root_rank = 0;
    }
    
    // default values
    if (global_count == 0) {
        global_count = 10 * 1024 * 1024; // totale globale di elementi
    }
    
    // Initialize OneCCL context (MPI, CCL, devices, communicator, logger)
    auto ctx = init_oneccl(output_dir, "scatter");
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
    
    // Validate root rank
    if (root_rank < 0 || root_rank >= size) {
        if (rank == 0) {
            std::cerr << "Invalid root rank " << root_rank << ". Must be between 0 and " << (size-1) << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    if (rank == 0) {
        std::cout << "Scattering from rank " << root_rank << " to " << size << " processes\n";
        std::cout << "Global elements: " << effective_global_count << " (" << local_count << " per rank)\n";
    }
     
    // dispatch based on dtype
    if (dtype == "int") {
        run_scatter<int>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "float") {
        run_scatter<float>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else if (dtype == "double") {
        run_scatter<double>(local_count, effective_global_count, size, rank, comm, q, stream, logger, dtype, root_rank);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    return 0;
}
