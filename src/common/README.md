# OneCCL Collectives Benchmarking - Refactored Structure

Questo progetto è stato refactorizzato per utilizzare una struttura modulare e riusabile per il benchmarking delle collettive OneCCL.

## Struttura del Progetto

```
src/
├── common/
│   ├── include/
│   │   ├── arg_parser.hpp              # Parser degli argomenti da riga di comando
│   │   ├── logger.hpp                  # Sistema di logging avanzato
│   │   ├── oneccl_setup.hpp           # Setup comune MPI/SYCL/OneCCL
│   │   ├── collective_benchmark.hpp    # Classe base template per collettive
│   │   ├── allreduce_benchmark.hpp    # Implementazione AllReduce
│   │   ├── allgather_benchmark.hpp    # Implementazione AllGather
│   │   └── broadcast_benchmark.hpp    # Implementazione Broadcast
│   └── src/
│       └── oneccl_setup.cpp           # Implementazione setup comune
└── oneccl/
    ├── allreduce/
    │   └── allreduce.cpp              # Main per AllReduce (refactorizzato)
    └── allgather/
        └── allgather.cpp              # Main per AllGather (esempio)
```

## Vantaggi della Refactorizzazione

### 1. **Eliminazione del Codice Duplicato**
- Setup MPI/SYCL/OneCCL centralizzato nella classe `OneCCLSetup`
- Template pattern per gestire timing, logging e verifica risultati
- Parser argomenti e logging riutilizzabili

### 2. **Facilità di Aggiunta di Nuove Collettive**
Per aggiungere una nuova collettiva bastano pochi passi:

1. Creare l'header della nuova collettiva (es. `reduce_benchmark.hpp`):
```cpp
#pragma once
#include "collective_benchmark.hpp"

template<typename T>
class ReduceBenchmark : public CollectiveBenchmark<T> {
    // Implementare i 5 metodi virtuali:
    // - allocate_buffers()
    // - initialize_buffers() 
    // - execute_collective()
    // - verify_results()
    // - cleanup_buffers()
};
```

2. Creare il main (es. `reduce.cpp`):
```cpp
#include "../../common/include/oneccl_setup.hpp"
#include "../../common/include/reduce_benchmark.hpp"
// ... template run_benchmark() e main() standard
```

### 3. **Manutenibilità**
- Bug fix e miglioramenti nel setup si propagano automaticamente
- Struttura consistente tra tutte le collettive
- Facilita testing e debugging

### 4. **Estensibilità**
- Facile aggiungere nuove feature (es. warm-up runs, multiple iterations)
- Possibilità di aggiungere nuovi tipi di verifica dei risultati
- Supporto per nuovi backend di comunicazione

## Come Compilare

### Setup Ambiente (Intel oneAPI)
```bash
source /opt/intel/oneapi/setvars.sh
```

### Compilazione AllReduce (esempio)
```bash
cd src/oneccl/allreduce
icpx -fsycl -I../../../src/common/include allreduce.cpp ../../../src/common/src/oneccl_setup.cpp -lmpi -lccl -o allreduce_benchmark
```

### Compilazione AllGather (esempio)  
```bash
cd src/oneccl/allgather
icpx -fsycl -I../../../src/common/include allgather.cpp ../../../src/common/src/oneccl_setup.cpp -lmpi -lccl -o allgather_benchmark
```

## Come Eseguire

```bash
# AllReduce
mpirun -n 4 ./allreduce_benchmark --dtype float --count 1000000 --output ./results

# AllGather
mpirun -n 4 ./allgather_benchmark --dtype int --count 500000 --output ./results
```

## Parametri Comuni

- `--dtype`: Tipo di dato (`int`, `float`, `double`)
- `--count`: Numero di elementi per rank (default: 10M)
- `--output`: Directory per i file di output CSV (opzionale)

## Aggiungere Nuove Collettive

### Esempio: Reduce

1. **Creare `reduce_benchmark.hpp`**:
```cpp
template<typename T>
class ReduceBenchmark : public CollectiveBenchmark<T> {
private:
    T* send_buf;
    T* recv_buf;  // solo per root
    int root_rank;

protected:
    void allocate_buffers() override {
        send_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        if (this->setup.rank == root_rank) {
            recv_buf = sycl::malloc_device<T>(this->count, this->setup.queue);
        }
    }
    
    void execute_collective() override {
        auto attr = ccl::create_operation_attr<ccl::reduce_attr>();
        ccl::reduce(send_buf, recv_buf, this->count, ccl::reduction::sum, 
                   root_rank, this->setup.comm, this->setup.stream, attr).wait();
    }
    
    // ... implementare altri metodi
};
```

2. **Creare `reduce.cpp`** seguendo il pattern degli altri main.

### Template Pattern dei Metodi Virtuali

Ogni benchmark deve implementare:

- **`allocate_buffers()`**: Alloca memoria device per i buffer
- **`initialize_buffers()`**: Inizializza i dati nei buffer  
- **`execute_collective()`**: Esegue la collettiva OneCCL
- **`verify_results()`**: Verifica la correttezza dei risultati
- **`cleanup_buffers()`**: Libera la memoria allocata

## Logging e Output

Il sistema di logging è già integrato e gestisce automaticamente:
- Timestamp, hostname, node info
- Metriche di performance (tempo di esecuzione)
- Output in formato CSV per analisi successive
- Run ID incrementali per distinguere esecuzioni multiple

## Prossimi Sviluppi

- [ ] Implementare Reduce, ReduceScatter, AllToAll
- [ ] Aggiungere supporto per multiple iterazioni con warm-up
- [ ] Integrare metriche di bandwidth e latenza
- [ ] Supporto per diverse configurazioni di messaggi (size sweep)
- [ ] Integrazione con sistemi di profiling (Intel VTune, etc.)
