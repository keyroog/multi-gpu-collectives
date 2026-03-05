// init_time.cu — Dedicated benchmark for MPI initialization time.
//
// Design:
//   - Designed to be invoked once per iteration from an external shell loop.
//   - Each invocation = one fresh mpirun process = true cold start.
//   - MPI_Init is the only initialization phase for MPI (unlike NCCL/oneCCL
//     where communicator creation is a separate timed phase).
//   - Timed with std::chrono (MPI_Wtime not available before MPI_Init).
//   - MPI_Barrier after init: timer stops when the LAST rank has finished.
//   - MPI_Gather of per-rank times to rank 0: rank 0 computes and logs the max.
//
// Usage (called from a shell loop):
//   mpirun -np 4 ./build/mpi/init_time --iter 1 --output results/...

#include <mpi.h>
#include <cuda_runtime.h>
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

using Clock = std::chrono::high_resolution_clock;

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

int main(int argc, char* argv[]) {
    // Time MPI_Init — must use chrono since MPI_Wtime is not available yet
    auto t0 = Clock::now();
    MPI_Init(&argc, &argv);
    auto t1 = Clock::now();
    double local_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Barrier so all ranks finish init before we gather times
    MPI_Barrier(MPI_COMM_WORLD);

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

    // GPU device: map each rank to its local GPU
    MPI_Comm local_comm;
    int local_rank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev == 0) {
        std::cerr << "[Rank " << rank << "] No CUDA devices found\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(local_rank % ndev);

    NodeInfo ni = gather_node_info(rank, size);

    // Gather per-rank times to rank 0
    std::vector<double> all_local_ms(size);
    MPI_Gather(&local_ms, 1, MPI_DOUBLE,
               all_local_ms.data(), 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // CSV setup
    std::ofstream csv;
    std::string csv_path;
    if (rank == 0 && !output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
        csv_path = output_dir + "/mpi_init_time_"
                 + std::to_string(size) + "ranks_results.csv";
        bool new_file = !std::filesystem::exists(csv_path);
        csv.open(csv_path, std::ios::app);
        if (new_file)
            csv << "timestamp,library,num_ranks,rank,hostname,"
                   "node_id,total_nodes,is_multi_node,iter,"
                   "local_init_ms,max_init_ms\n";
    }

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
                csv << ts << ",mpi," << size << "," << r << ","
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
