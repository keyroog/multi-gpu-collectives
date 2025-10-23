// pingpong_two_gpus.cpp
#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const size_t msg_bytes = (argc > 1) ? std::stoul(argv[1]) : (1 << 20);
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

    // 1) Trova almeno 2 root GPU nel sistema
    sycl::platform plat{sycl::gpu_selector_v};
    auto gpus = plat.get_devices(sycl::info::device_type::gpu);
    if (gpus.size() < 2) {
        if (rank == 0) std::fprintf(stderr, "Servono almeno 2 GPU sul nodo\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Context multi-root (esplicito nella guida) e queue per la GPU del rank
    std::vector<sycl::device> used{gpus[0], gpus[1]};
    sycl::context ctx{used};
    sycl::device my_dev = used[rank];
    sycl::queue q{ctx, my_dev};

    // 2) oneCCL communicator/stream (pattern Sample Application)
    auto dev = ccl::create_device(q.get_device());
    auto cctx = ccl::create_context(q.get_context());

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

    // 3) USM device buffer
    const size_t count = msg_bytes;
    using u8 = std::uint8_t;
    u8* buf = static_cast<u8*>(sycl::malloc_device(count, q));
    if (!buf) {
        std::fprintf(stderr, "[%d] malloc_device fallita\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    q.fill<u8>(buf, 0x5A, count).wait();

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

    // 4) Misure
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
        double ms_rank0 = (rank == 0) ? ms : 0.0;
        MPI_Reduce(rank == 0 ? &ms_rank0 : &ms, &ms_rank0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) total_ms += ms_rank0;
    }

    if (rank == 0) {
        double avg_ms = total_ms / iters;
        double rtt_gbps = (2.0 * msg_bytes) / (avg_ms * 1e6);
        std::printf("[TWO GPUs] msg=%zu B iters=%d  avg RTT=%.3f ms  throughput≈%.3f GB/s\n",
                    msg_bytes, iters, avg_ms, rtt_gbps);
    }

    sycl::free(buf, q);
    MPI_Finalize();
    return 0;
}