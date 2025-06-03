#pragma once

#include "collective_benchmark.hpp"

template<typename T>
class BroadcastBenchmark : public CollectiveBenchmark<T> {
private:
    T* buffer;
    int root_rank;
    
public:
    BroadcastBenchmark(size_t count, OneCCLSetup::SetupResult setup, Logger& logger, 
                      const std::string& data_type, int root_rank = 0)
        : CollectiveBenchmark<T>(count, setup, logger, "broadcast", data_type), 
          buffer(nullptr), root_rank(root_rank) {}
    
protected:
    void allocate_buffers() override {
        buffer = sycl::malloc_device<T>(this->count, this->setup.queue);
        
        if (!buffer) {
            throw std::runtime_error("Failed to allocate device buffer");
        }
    }
    
    void initialize_buffers() override {
        this->setup.queue.submit([&](auto& h) {
            h.parallel_for(this->count, [=](auto id) {
                if (this->setup.rank == root_rank) {
                    // Root rank initializes with meaningful data
                    buffer[id] = static_cast<T>(root_rank * 1000 + id);
                } else {
                    // Other ranks initialize with -1 
                    buffer[id] = static_cast<T>(-1);
                }
            });
        });
    }
    
    void execute_collective() override {
        auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
        ccl::broadcast(buffer, this->count, root_rank,
                      this->setup.comm, this->setup.stream, attr).wait();
    }
    
    void verify_results() override {
        sycl::buffer<T> check_buf(this->count);
        this->setup.queue.submit([&](auto& h) {
            sycl::accessor acc(check_buf, h, sycl::write_only);
            h.parallel_for(this->count, [=](auto id) {
                T expected_value = static_cast<T>(root_rank * 1000 + id);
                
                if (buffer[id] != expected_value) {
                    acc[id] = static_cast<T>(-1);  // Mark as error
                } else {
                    acc[id] = static_cast<T>(0);   // Mark as correct
                }
            });
        });
        this->setup.queue.wait_and_throw();
        
        // Check results on host
        sycl::host_accessor acc(check_buf, sycl::read_only);
        size_t i = 0;
        for (; i < this->count; ++i) {
            if (acc[i] == static_cast<T>(-1)) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == this->count) {
            std::cout << "PASSED\n";
        }
    }
    
    void cleanup_buffers() override {
        if (buffer) {
            sycl::free(buffer, this->setup.queue);
            buffer = nullptr;
        }
    }
};
