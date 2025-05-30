#include <iostream>
#include <mpi.h>
#include "oneapi/ccl.hpp"
#include <chrono>                    // <<< aggiunto
#include "common/include/arg_parser.hpp"
#include "common/include/collective_runner.hpp"
#include <string>

void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
        MPI_Finalize();
    }
}

int main(int argc, char* argv[]) {
    return collective_runner::main_collective(argc, argv, [&](int argc, char** argv) {
        using namespace collective_runner;
        // fetch parsed args
        ArgParser parser(argc, argv);
        std::string dtype = parser.get<std::string>("--dtype");
        size_t count = parser.get<size_t>("--count");
        // obtain CCL communicator and SYCL queue injected by main_collective
        auto& comm = /*...provided by context...*/;
        auto& q = /*...provided by context...*/;
        // dispatch to template runner for allreduce
        if (dtype == "int") run_collective<int>(count, size, rank, comm, q, 
            [&](int* s, int* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                ccl::allreduce(s, r, c, ccl::reduction::sum, cm, qu, deps).wait();
            });
        else if (dtype == "float") run_collective<float>(count, size, rank, comm, q, 
            [&](float* s, float* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                ccl::allreduce(s, r, c, ccl::reduction::sum, cm, qu, deps).wait();
            });
        else if (dtype == "double") run_collective<double>(count, size, rank, comm, q, 
            [&](double* s, double* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                ccl::allreduce(s, r, c, ccl::reduction::sum, cm, qu, deps).wait();
            });
        else { std::cerr << "Unsupported dtype: " << dtype << std::endl; exit(-1);} 
    });
}