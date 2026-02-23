// init_time.cpp — Dedicated benchmark for oneCCL communicator initialization time.
//
// Design:
//   - Designed to be invoked once per iteration from an external shell loop.
//   - Each invocation = one fresh mpirun process = true cold start.
//
//   Phase lib_init (printed, not CSV):
//     ccl::init() is a global one-time call with no NCCL equivalent. It must run
//     before MPI_Init. Timed with std::chrono, max printed for info only.
//
//   Phase comm_init (timed and logged to CSV):
//     MPI_Barrier → KVS creation + MPI_Bcast addr + create_communicator +
//     create_stream → MPI_Barrier → measure.
//     Warmup block (untimed): one cycle before the measurement.
//
// Usage (called from run_init_time.sh):
//   mpirun -np 4 ./build/oneccl/init_time --iter 1 --output results/...

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <unistd.h>
#include <filesystem>

// ---- Utilities ----------------------------------------------------------------

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    return ss.str();
}

struct NodeInfo {
    std::vector<std::string> rank_hostnames;
    std::vector<int>         rank_node_ids;
    int                      total_nodes;
    bool                     is_multi_node;
};

static NodeInfo gather_node_info(int rank, int size) {
    const int MAX_HOST = 256;
    char local_host[MAX_HOST] = {};
    gethostname(local_host, MAX_HOST - 1);

    std::vector<char> all_hosts(size * MAX_HOST, '\0');
    MPI_Allgather(local_host, MAX_HOST, MPI_CHAR,
                  all_hosts.data(), MAX_HOST, MPI_CHAR, MPI_COMM_WORLD);

    std::vector<std::string> hostnames(size);
    for (int i = 0; i < size; i++)
        hostnames[i] = std::string(all_hosts.data() + i * MAX_HOST);

    std::unordered_set<std::string> unique_set(hostnames.begin(), hostnames.end());
    std::vector<std::string> sorted_unique(unique_set.begin(), unique_set.end());
    std::sort(sorted_unique.begin(), sorted_unique.end());

    int total_nodes = (int)sorted_unique.size();

    std::vector<int> node_ids(size);
    for (int r = 0; r < size; r++) {
        for (int i = 0; i < (int)sorted_unique.size(); i++) {
            if (sorted_unique[i] == hostnames[r]) { node_ids[r] = i; break; }
        }
    }

    return {hostnames, node_ids, total_nodes, total_nodes > 1};
}

static sycl::device pick_device(int local_rank) {
    std::vector<sycl::device> gpus;
    for (const auto& plat : sycl::platform::get_platforms()) {
        for (const auto& dev : plat.get_devices()) {
            if (dev.is_gpu()) gpus.push_back(dev);
        }
    }
    if (local_rank >= (int)gpus.size())
        throw std::runtime_error("Not enough GPU devices for local rank "
                                 + std::to_string(local_rank));
    return gpus[local_rank];
}

// ---- Main --------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // ccl::init() must precede MPI_Init — timed for info, not logged to CSV
    using Clock = std::chrono::high_resolution_clock;
    auto lib_t0 = Clock::now();
    ccl::init();
    auto lib_t1 = Clock::now();
    double lib_init_ms = std::chrono::duration<double, std::milli>(lib_t1 - lib_t0).count();

    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse args
    int iter_num = 1;
    std::string output_dir;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--iter" && i + 1 < argc)
            iter_num = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }

    // GPU device
    MPI_Comm local_comm;
    int local_rank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    sycl::device dev = pick_device(local_rank);
    sycl::context sycl_ctx(dev);
    sycl::queue q(sycl_ctx, dev, {sycl::property::queue::in_order()});

    NodeInfo ni = gather_node_info(rank, size);

    // Print lib_init info (max across ranks, for reference only)
    {
        std::vector<double> all_lib_ms(size);
        MPI_Gather(&lib_init_ms, 1, MPI_DOUBLE,
                   all_lib_ms.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double max_lib = *std::max_element(all_lib_ms.begin(), all_lib_ms.end());
            std::cout << "  [info] ccl::init()  max=" << std::fixed
                      << std::setprecision(3) << max_lib << " ms\n";
        }
    }

    // CSV setup — rank 0 owns the file, appends; writes header if new
    std::ofstream csv;
    std::string csv_path;
    if (rank == 0 && !output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
        csv_path = output_dir + "/oneccl_init_time_"
                 + std::to_string(size) + "ranks_results.csv";
        bool new_file = !std::filesystem::exists(csv_path);
        csv.open(csv_path, std::ios::app);
        if (new_file)
            csv << "timestamp,library,num_ranks,rank,hostname,"
                   "node_id,total_nodes,is_multi_node,iter,"
                   "local_init_ms,max_init_ms\n";
    }

    // ---- Warmup (not timed, not logged) ----------------------------------------
    {
        MPI_Barrier(MPI_COMM_WORLD);
        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type addr;
        if (rank == 0) { kvs = ccl::create_main_kvs(); addr = kvs->get_address(); }
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) kvs = ccl::create_kvs(addr);
        auto ccl_dev = ccl::create_device(q.get_device());
        auto ccl_ctx = ccl::create_context(q.get_context());
        auto comm    = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
        auto stream  = ccl::create_stream(q);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ---- Single measurement (KVS + communicator + stream) ----------------------
    std::vector<double> all_local_ms(size);
    double local_ms = 0.0;

    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type addr;
        if (rank == 0) { kvs = ccl::create_main_kvs(); addr = kvs->get_address(); }
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) kvs = ccl::create_kvs(addr);

        auto ccl_dev = ccl::create_device(q.get_device());
        auto ccl_ctx = ccl::create_context(q.get_context());
        auto comm    = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
        auto stream  = ccl::create_stream(q);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();
        local_ms = (t_end - t_start) * 1000.0;
        // CCL objects destroyed here (RAII)
    }

    MPI_Gather(&local_ms, 1, MPI_DOUBLE,
               all_local_ms.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double max_ms = *std::max_element(all_local_ms.begin(), all_local_ms.end());

        std::cout << "  [" << std::setw(2) << iter_num << "]"
                  << "  max=" << std::fixed << std::setprecision(3)
                  << std::setw(8) << max_ms << " ms"
                  << "  per-rank: [";
        for (int r = 0; r < size; r++) {
            std::cout << std::fixed << std::setprecision(3) << all_local_ms[r];
            if (r < size - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        if (csv.is_open()) {
            std::string ts = get_timestamp();
            for (int r = 0; r < size; r++) {
                csv << ts << ",oneccl," << size << "," << r << ","
                    << ni.rank_hostnames[r] << ","
                    << ni.rank_node_ids[r] << ","
                    << ni.total_nodes << ","
                    << (ni.is_multi_node ? "true" : "false") << ","
                    << iter_num << ","
                    << std::fixed << std::setprecision(3) << all_local_ms[r] << ","
                    << std::fixed << std::setprecision(3) << max_ms << "\n";
            }
            csv.close();
        }
    }

    MPI_Finalize();
    return 0;
}
