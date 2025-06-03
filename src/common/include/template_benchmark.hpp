#pragma once

#include "collective_benchmark.hpp"

// Template per implementare una nuova collettiva
// Sostituire "Template" con il nome della collettiva (es. "Reduce", "AllToAll", etc.)

template<typename T>
class TemplateBenchmark : public CollectiveBenchmark<T> {
private:
    // Dichiarare qui i buffer device necessari
    T* send_buf;
    T* recv_buf;
    
    // Aggiungere eventuali parametri specifici della collettiva
    // int root_rank;  // per collettive root-based come Reduce, Broadcast
    
public:
    TemplateBenchmark(size_t count, OneCCLSetup::SetupResult setup, Logger& logger, const std::string& data_type)
        : CollectiveBenchmark<T>(count, setup, logger, "template", data_type),  // Sostituire "template" con nome collettiva
          send_buf(nullptr), recv_buf(nullptr) {}
    
    // Aggiungere costruttori con parametri aggiuntivi se necessario
    // TemplateBenchmark(..., int root_rank = 0) : ..., root_rank(root_rank) {}
    
protected:
    void allocate_buffers() override {
        // Allocare i buffer device necessari
        send_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        recv_buf = sycl::malloc_device<T>(this->count, this->setup.queue);  // Adattare dimensione se necessario
        
        if (!send_buf || !recv_buf) {
            throw std::runtime_error("Failed to allocate device buffers");
        }
    }
    
    void initialize_buffers() override {
        // Inizializzare i buffer con dati di test
        this->setup.queue.submit([&](auto& h) {
            h.parallel_for(this->count, [=](auto id) {
                send_buf[id] = static_cast<T>(this->setup.rank * 1000 + id);
                recv_buf[id] = static_cast<T>(-1);  // Valore di default
            });
        });
    }
    
    void execute_collective() override {
        // Eseguire la collettiva OneCCL
        // Esempi:
        
        // Per AllReduce:
        // auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
        // ccl::allreduce(send_buf, recv_buf, this->count, ccl::reduction::sum, 
        //               this->setup.comm, this->setup.stream, attr).wait();
        
        // Per AllGather:
        // auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        // ccl::allgather(send_buf, recv_buf, this->count, 
        //               this->setup.comm, this->setup.stream, attr).wait();
        
        // Per Broadcast:
        // auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
        // ccl::broadcast(send_buf, this->count, root_rank,
        //               this->setup.comm, this->setup.stream, attr).wait();
        
        // Per Reduce:
        // auto attr = ccl::create_operation_attr<ccl::reduce_attr>();
        // ccl::reduce(send_buf, recv_buf, this->count, ccl::reduction::sum, 
        //            root_rank, this->setup.comm, this->setup.stream, attr).wait();
        
        // IMPLEMENTARE QUI LA COLLETTIVA SPECIFICA
        throw std::runtime_error("Template collective not implemented");
    }
    
    void verify_results() override {
        // Verificare la correttezza dei risultati
        sycl::buffer<T> check_buf(this->count);  // Adattare dimensione se necessario
        this->setup.queue.submit([&](auto& h) {
            sycl::accessor acc(check_buf, h, sycl::write_only);
            h.parallel_for(this->count, [=](auto id) {
                // Calcolare il valore atteso basato sulla semantica della collettiva
                T expected_value = static_cast<T>(0);  // IMPLEMENTARE CALCOLO VALORE ATTESO
                
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
        for (; i < this->count; ++i) {  // Adattare dimensione se necessario
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
        // Liberare i buffer allocati
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

/*
ISTRUZIONI PER L'USO:

1. Copiare questo file e rinominarlo (es. reduce_benchmark.hpp)

2. Sostituire "Template" con il nome della collettiva:
   - TemplateBenchmark -> ReduceBenchmark
   - "template" -> "reduce"

3. Implementare i metodi:
   - allocate_buffers(): Allocare i buffer necessari
   - initialize_buffers(): Inizializzare con dati di test
   - execute_collective(): Chiamare la funzione OneCCL appropriata
   - verify_results(): Verificare la correttezza
   - cleanup_buffers(): Liberare memoria

4. Creare il main seguendo il pattern degli altri esempi

5. Aggiungere target al Makefile per la compilazione

Esempi di riferimento:
- allreduce_benchmark.hpp per collettive di riduzione
- allgather_benchmark.hpp per collettive di raccolta
- broadcast_benchmark.hpp per collettive di broadcast
*/
