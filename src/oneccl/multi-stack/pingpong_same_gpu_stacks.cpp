// pingpong_same_gpu_stacks.cpp
#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    // Parametri: dimensione messaggio in byte, iterazioni
    const size_t msg_bytes = (argc > 1) ? std::stoul(argv[1]) : (1 << 20); // 1 MiB
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    MPI_Init(&argc, &argv);
    ccl::init();

    int world = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world != 2) {
        if (rank == 0) std::fprintf(stderr, "Questo test richiede esattamente 2 processi (mpiexec -n 2)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 1) Seleziona una GPU e splitta in subdevices (stack) via affinity_domain::numa (COMPOSITE mode)
    sycl::device root_gpu{sycl::gpu_selector_v};

    auto part_prop = root_gpu.get_info<sycl::info::device::partition_properties>();
    if (part_prop.empty()) {
        std::fprintf(stderr, "Il dispositivo selezionato non supporta il partizionamento in subdevices.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    std::vector<sycl::device> stacks =
        root_gpu.create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);

    if (stacks.size() < 2) {
        if (rank == 0) std::fprintf(stderr, "La GPU selezionata non espone almeno 2 stack/subdevices.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Prendiamo i primi due stack e creiamo un context condiviso (Explicit Scaling - SYCL)
    std::vector<sycl::device> used{stacks[0], stacks[1]};
    sycl::context ctx{used};

    // Ogni rank usa un subdevice diverso
    sycl::device my_dev = used[rank];
    sycl::queue q{ctx, my_dev};

    // 2) Costruisci oneCCL communicator/stream in stile Sample Application
    auto dev = ccl::create_device(q.get_device());
    auto cctx = ccl::create_context(q.get_context());

    // KVS bootstrap via MPI Bcast (pattern sample app)
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
    }
    MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) kvs = ccl::create_kvs(main_addr);

    auto comm = ccl::create_communicator(world, rank, dev, cctx, kvs);
    auto stream = ccl::create_stream(q);

    // 3) Allochiamo USM device e prepariamo il buffer
    const size_t count = msg_bytes; // usiamo int8 per contare "byte"
    using u8 = std::uint8_t;

    u8* buf = static_cast<u8*>(sycl::malloc_device(count, q));
    if (!buf) {
        std::fprintf(stderr, "[%d] malloc_device fallita\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    // inizializza (opzionale)
    q.fill<u8>(buf, 0xA5, count).wait();

    const int peer = 1 - rank;

    // Warmup
    if (rank == 0) {
        ccl::send(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
        ccl::recv(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
    } else {
        ccl::recv(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
        ccl::send(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 4) Misura ping-pong (round-trip) con chrono
    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            ccl::send(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
            ccl::recv(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
        } else {
            ccl::recv(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
            ccl::send(buf, count, ccl::datatype::int8, peer, comm, stream).wait();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Media calcolata solo su rank 0
        double ms_rank0 = (rank == 0) ? ms : 0.0;
        MPI_Reduce(rank == 0 ? &ms_rank0 : &ms, &ms_rank0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) total_ms += ms_rank0;
    }

    if (rank == 0) {
        double avg_ms = total_ms / iters;
        double rtt_gbps = (2.0 * msg_bytes) / (avg_ms * 1e6); // invio+ricezione => 2x bytes; GBps (base 10)
        std::printf("[SAME GPU/STACKS] msg=%zu B iters=%d  avg RTT=%.3f ms  throughput≈%.3f GB/s\n",
                    msg_bytes, iters, avg_ms, rtt_gbps);
    }

    sycl::free(buf, q);
    MPI_Finalize();
    return 0;
}