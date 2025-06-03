#include "../../common/include/arg_parser.hpp"
#include "../../common/include/logger.hpp"
#include "../../common/include/oneccl_setup.hpp"
#include "../../common/include/allgather_benchmark.hpp"
#include <string>

template<typename T>
void run_benchmark(size_t count, const std::string& data_type, Logger& logger) {
    OneCCLSetup setup;
    auto result = setup.initialize();
    
    AllGatherBenchmark<T> benchmark(count, result, logger, data_type);
    benchmark.run();
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

    // Crea il logger
    Logger logger(output_dir, "oneccl", "allgather");
    
    // dispatch based on dtype
    if (dtype == "int") {
        run_benchmark<int>(count, dtype, logger);
    } else if (dtype == "float") {
        run_benchmark<float>(count, dtype, logger);
    } else if (dtype == "double") {
        run_benchmark<double>(count, dtype, logger);
    } else {
        std::cerr << "Unsupported dtype: " << dtype << std::endl;
        exit(-1);
    }
    
    return 0;
}
