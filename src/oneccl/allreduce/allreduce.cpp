#include <iostream>
#include <mpi.h>
#include "oneapi/ccl.hpp"
#include <chrono>                    // <<< aggiunto
#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include <string>

// Template wrapper for different data types
template <typename T>
double run_allreduce(size_t count, int size, int rank, ccl::communicator& comm, sycl::queue& q, ccl::stream stream, 
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
    
    // Log dei risultati individuali
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
    
    return elapsed_ms;  // Ritorna il tempo per la raccolta delle statistiche
}
        for (; i < count; ++i) if (acc[i] == static_cast<T>(-1)) { std::cout << "FAILED\n"; break; }
        if (i == count) std::cout << "PASSED\n";
    }
    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);
}

void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
        MPI_Finalize();
    }
}

int main(int argc, char* argv[]) {
    ArgParser parser(argc, argv);
    parser.add<std::string>("--dtype").add<size_t>("--count").add<std::string>("--output");
    parser.parse();
    std::string dtype = parser.get<std::string>("--dtype");
    size_t count = parser.get<size_t>("--count");
    std::string output_dir = parser.get<std::string>("--output");
    
    //default value for count
    if (count == 0) {
        count = 10 * 1024 * 1024; // Default value if not provided
    }

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    // Inizio di una nuova esecuzione completa del benchmark (solo rank 0)
    if (rank == 0) {
        Logger::start_new_run();
        std::cout << "=== Starting benchmark run #" << Logger::get_current_run_id() << " ===" << std::endl;
    }
    
    // Sincronizza tutti i ranks prima di procedere
    MPI_Barrier(MPI_COMM_WORLD);

    /* find and initialize Level-Zero devices and queues */
    std::vector<sycl::device> devices;
    std::vector<sycl::queue> queues;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto &platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
        if (is_level_zero) {
            std::cout << "Platform_name is:  " << platform_name << std::endl;
            auto device_list = platform.get_devices();
            for (const auto &device : device_list) {
                if (device.is_gpu()) {
                    devices.push_back(device);
                }
            }
        }
    }

    if (devices.size() < size) {
        std::cerr << "Not enough devices for all ranks" << std::endl;
        exit(-1);
    }

    sycl::context context(devices);
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i == rank) { /* Only create a queue for the current rank's device */
            queues.push_back(sycl::queue(context, devices[i], {sycl::property::queue::in_order()}));
            break;
        }
    }

    if (queues.empty()) {
        std::cerr << "No queue created for rank " << rank << std::endl;
        exit(-1);
    }

    /* Use the only queue in the queues vector for the current rank */
    sycl::queue& q = queues[0];

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);
    
    // Crea il logger
    Logger logger(output_dir, "oneccl", "allreduce");
    
    // Log GPU topology information (only rank 0 to avoid spam)
    if (rank == 0) {
        logger.log_gpu_topology_info(rank);
    }
    
    double my_time = 0.0;
    
    // dispatch based on dtype
    if (dtype == "int") {
        my_time = run_allreduce<int>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "float") {
        my_time = run_allreduce<float>(count, size, rank, comm, q, stream, logger, dtype);
    } else if (dtype == "double") {
        my_time = run_allreduce<double>(count, size, rank, comm, q, stream, logger, dtype);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    // Raccolta delle statistiche collettive per calcolare goodput
    std::vector<double> all_times(size);
    MPI_Allgather(&my_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Calcola le statistiche collettive e mostra il goodput (solo rank 0)
    if (rank == 0) {
        Logger::CollectiveStats stats = Logger::calculate_collective_stats(all_times);
        
        // Log del summary tradizionale
        logger.log_summary(dtype, count, size, stats.min_time_ms, stats.max_time_ms, stats.avg_time_ms);
        
        // Log delle statistiche avanzate con analisi della varianza
        std::string context = "Single-Node";
        if (size > 8) {  // Euristica: più di 8 ranks suggerisce scenario multi-nodo
            context = "Multi-Node";
        }
        logger.log_collective_stats(dtype, count, stats, context);
        
        std::cout << "\n=== GOODPUT ANALYSIS ===" << std::endl;
        std::cout << "Worst-rank time (Goodput): " << std::fixed << std::setprecision(3) << stats.goodput_ms << " ms" << std::endl;
        std::cout << "Best-rank time: " << std::fixed << std::setprecision(3) << stats.min_time_ms << " ms" << std::endl;
        std::cout << "Performance variation: " << std::fixed << std::setprecision(1) 
                  << ((stats.max_time_ms - stats.min_time_ms) / stats.avg_time_ms * 100.0) << "%" << std::endl;
        
        if (stats.max_time_ms > stats.min_time_ms * 1.1) {  // > 10% variation
            std::cout << "WARNING: High performance variation detected!" << std::endl;
            std::cout << "  This may indicate load imbalance, network issues, or topology problems." << std::endl;
        }
    }
    
    // Final synchronization and summary (rank 0 only)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "=== Completed benchmark run #" << Logger::get_current_run_id() << " ===" << std::endl;
    }
    
    return 0;
}