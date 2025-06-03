#pragma once

#include "oneccl_setup.hpp"
#include "logger.hpp"
#include <string>
#include <chrono>
#include <iomanip>

template<typename T>
class CollectiveBenchmark {
protected:
    size_t count;
    OneCCLSetup::SetupResult setup;
    Logger& logger;
    std::string collective_name;
    std::string data_type;
    
public:
    CollectiveBenchmark(size_t count, OneCCLSetup::SetupResult setup, 
                       Logger& logger, const std::string& collective_name, 
                       const std::string& data_type)
        : count(count), setup(setup), logger(logger), 
          collective_name(collective_name), data_type(data_type) {}
    
    virtual ~CollectiveBenchmark() = default;
    
    // Template method pattern - main execution flow
    void run() {
        allocate_buffers();
        initialize_buffers();
        
        auto start = std::chrono::high_resolution_clock::now();
        execute_collective();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        logger.log_result(data_type, count, setup.size, setup.rank, elapsed_ms);
        print_timing(elapsed_ms);
        
        verify_results();
        cleanup_buffers();
    }
    
protected:
    // Virtual methods to be implemented by specific collectives
    virtual void allocate_buffers() = 0;
    virtual void initialize_buffers() = 0;
    virtual void execute_collective() = 0;
    virtual void verify_results() = 0;
    virtual void cleanup_buffers() = 0;
    
    void print_timing(double elapsed_ms) {
        std::cout << "Rank " << setup.rank << " " << collective_name 
                  << " time: " << std::fixed << std::setprecision(3) 
                  << elapsed_ms << " ms\n";
    }
};
