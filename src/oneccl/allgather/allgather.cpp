#include "common/include/collective_runner.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    return collective_runner::main_collective(argc, argv,
        [](const std::string& dtype,
           const std::string& count_str,
           size_t count,
           const std::string& logpath,
           int size,
           int rank,
           ccl::communicator& comm,
           sycl::queue& q) {
            if (dtype == "int") {
                collective_runner::run_collective<int>("allgather", dtype, count, size, rank, comm, q, logpath,
                    [&](int* s, int* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                        ccl::allgather(s, r, c, cm, qu, deps).wait();
                    });
            } else if (dtype == "float") {
                collective_runner::run_collective<float>("allgather", dtype, count, size, rank, comm, q, logpath,
                    [&](float* s, float* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                        ccl::allgather(s, r, c, cm, qu, deps).wait();
                    });
            } else if (dtype == "double") {
                collective_runner::run_collective<double>("allgather", dtype, count, size, rank, comm, q, logpath,
                    [&](double* s, double* r, size_t c, ccl::communicator& cm, sycl::queue& qu, const std::vector<ccl::event>& deps) {
                        ccl::allgather(s, r, c, cm, qu, deps).wait();
                    });
            } else {
                std::cerr << "Unsupported dtype: " << dtype << std::endl;
                exit(-1);
            }
        });
}
