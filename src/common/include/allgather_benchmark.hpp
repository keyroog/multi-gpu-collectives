#pragma once

#include "collective_benchmark.hpp"

template<typename T>
class AllGatherBenchmark : public CollectiveBenchmark<T> {
private:
    T* send_buf;
    T* recv_buf;
    
public:
    AllGatherBenchmark(size_t count, OneCCLSetup::SetupResult setup, Logger& logger, const std::string& data_type)
        : CollectiveBenchmark<T>(count, setup, logger, "allgather", data_type), 
          send_buf(nullptr), recv_buf(nullptr) {}
    
protected:
    void allocate_buffers() override {
        send_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        recv_buf = sycl::malloc_device<T>(this->count * this->setup.size, this->setup.queue);
        
        if (!send_buf || !recv_buf) {
            throw std::runtime_error("Failed to allocate device buffers");
        }
    }
    
    void initialize_buffers() override {
        this->setup.queue.submit([&](auto& h) {
            h.parallel_for(this->count, [=](auto id) {
                send_buf[id] = static_cast<T>(this->setup.rank * 1000 + id);
            });
        });
        
        this->setup.queue.submit([&](auto& h) {
            h.parallel_for(this->count * this->setup.size, [=](auto id) {
                recv_buf[id] = static_cast<T>(-1);
            });
        });
    }
    
    void execute_collective() override {
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        ccl::allgather(send_buf, recv_buf, this->count, 
                      this->setup.comm, this->setup.stream, attr).wait();
    }
    
    void verify_results() override {
        sycl::buffer<T> check_buf(this->count * this->setup.size);
        this->setup.queue.submit([&](auto& h) {
            sycl::accessor acc(check_buf, h, sycl::write_only);
            h.parallel_for(this->count * this->setup.size, [=](auto id) {
                int rank_source = id / this->count;
                int element_idx = id % this->count;
                T expected_value = static_cast<T>(rank_source * 1000 + element_idx);
                
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
        for (; i < this->count * this->setup.size; ++i) {
            if (acc[i] == static_cast<T>(-1)) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == this->count * this->setup.size) {
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
