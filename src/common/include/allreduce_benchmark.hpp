#pragma once

#include "collective_benchmark.hpp"

template<typename T>
class AllReduceBenchmark : public CollectiveBenchmark<T> {
private:
    T* send_buf;
    T* recv_buf;
    T expected_sum;
    
public:
    AllReduceBenchmark(size_t count, OneCCLSetup::SetupResult setup, Logger& logger, const std::string& data_type)
        : CollectiveBenchmark<T>(count, setup, logger, "allreduce", data_type), 
          send_buf(nullptr), recv_buf(nullptr) {}
    
protected:
    void allocate_buffers() override {
        send_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        recv_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        
        if (!send_buf || !recv_buf) {
            throw std::runtime_error("Failed to allocate device buffers");
        }
    }
    
    void initialize_buffers() override {
        auto e = this->setup.queue.submit([&](auto& h) {
            h.parallel_for(this->count, [=](auto id) {
                send_buf[id] = static_cast<T>(this->setup.rank + id + 1);
                recv_buf[id] = static_cast<T>(-1);
            });
        });
        
        // Compute expected sum for verification
        expected_sum = static_cast<T>(0);
        for (int i = 1; i <= this->setup.size; ++i) {
            expected_sum += static_cast<T>(i);
        }
    }
    
    void execute_collective() override {
        // Create dependencies from buffer initialization
        std::vector<ccl::event> deps;
        
        auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
        ccl::allreduce(send_buf, recv_buf, this->count, ccl::reduction::sum, 
                      this->setup.comm, this->setup.stream, attr, deps).wait();
    }
    
    void verify_results() override {
        sycl::buffer<T> check_buf(this->count);
        this->setup.queue.submit([&](auto& h) {
            sycl::accessor acc(check_buf, h, sycl::write_only);
            h.parallel_for(this->count, [=](auto id) {
                T expected_value = static_cast<T>(expected_sum + this->setup.size * id);
                if (recv_buf[id] != expected_value) {
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
        if (send_buf) {
            sycl::free(send_buf, this->setup.queue);
            send_buf = nullptr;
        }
        if (recv_buf) {
            sycl::free(recv_buf, this->setup.queue);
            recv_buf = nullptr;
        }
    }
};
