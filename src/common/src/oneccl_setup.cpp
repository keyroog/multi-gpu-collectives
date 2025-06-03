#include "../include/oneccl_setup.hpp"
#include <cstdlib>

OneCCLSetup::OneCCLSetup() : size(0), rank(0), initialized(false) {}

OneCCLSetup::~OneCCLSetup() {
    // Il cleanup di MPI viene gestito da atexit
}

OneCCLSetup::SetupResult OneCCLSetup::initialize() {
    if (initialized) {
        throw std::runtime_error("OneCCLSetup already initialized");
    }
    
    // Initialize CCL
    ccl::init();
    
    // Setup MPI
    setup_mpi();
    
    // Setup SYCL devices
    setup_sycl_devices();
    
    // Get queue reference
    sycl::queue& q = queues[0];
    
    // Setup CCL communicator and stream
    ccl::communicator comm;
    ccl::stream stream;
    setup_ccl_communicator(q, comm, stream);
    
    initialized = true;
    
    return SetupResult{q, comm, stream, size, rank};
}

void OneCCLSetup::setup_mpi() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    atexit(mpi_finalize);
}

void OneCCLSetup::setup_sycl_devices() {
    /* find and initialize Level-Zero devices and queues */
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
}

void OneCCLSetup::setup_ccl_communicator(sycl::queue& q, ccl::communicator& comm, ccl::stream& stream) {
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
    comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    stream = ccl::create_stream(q);
}

void OneCCLSetup::mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
        MPI_Finalize();
    }
}
