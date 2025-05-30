# OneCCL AllReduce Benchmark con Sistema di Logging

## Panoramica

Questo programma implementa un benchmark per l'operazione AllReduce usando Intel OneCCL, con un sistema di logging avanzato per analizzare le performance.

## Funzionalità del Sistema di Logging

### 1. Classe Logger (`logger.hpp`)

La classe `Logger` fornisce:
- **Logging strutturato**: Salva i risultati in formato CSV con tutte le informazioni rilevanti
- **Gestione automatica delle directory**: Crea automaticamente le directory di output
- **Timestamp automatici**: Ogni log include un timestamp preciso
- **Separazione per data type**: File CSV separati per ogni tipo di dato
- **Header automatici**: Aggiunge automaticamente gli header ai nuovi file CSV

### 2. Formato dei Dati

I file CSV contengono le seguenti colonne:
```csv
timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,time_ms
```

Esempio:
```csv
20250530_142051,oneccl,allreduce,int,4096,1024,2,0,5.234
20250530_142051,oneccl,allreduce,int,4096,1024,2,1,4.876
```

## Compilazione

```bash
# Assicurati di aver caricato l'ambiente Intel OneAPI
source /opt/intel/oneapi/setvars.sh

# Compila il programma
icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl
```

## Utilizzo

### Esecuzione Singola

```bash
# Esecuzione con logging
mpirun -n 2 ./allreduce --dtype int --count 1024 --output ../../../results/oneccl

# Esecuzione senza logging (solo console)
mpirun -n 2 ./allreduce --dtype int --count 1024
```

### Parametri

- `--dtype`: Tipo di dato (`int`, `float`, `double`)
- `--count`: Numero di elementi nel messaggio (default: 10M se 0)
- `--output`: Directory per salvare i risultati CSV (opzionale)

### Benchmark Automatico

Usa lo script fornito per testare automaticamente diverse combinazioni:

```bash
# Esegui il benchmark completo
./run_benchmark.sh
```

Lo script testa:
- **Data types**: int, float, double
- **Message sizes**: da 1024 a 16M elementi
- **Output**: Salva tutti i risultati in `../results/oneccl`

## Generazione Grafici

### Prerequisiti Python

```bash
pip install pandas matplotlib seaborn
```

### Generazione

```bash
# Genera tutti i grafici dai risultati
python3 ../../../scripts/generate_plots.py \
    --input ../../../results/oneccl \
    --output ../../../results/plots
```

### Tipi di Grafici Generati

1. **Performance vs Message Size**: Tempo di esecuzione in funzione della dimensione del messaggio
2. **Performance by Data Type**: Confronto tra diversi tipi di dato
3. **Scaling Analysis**: Analisi della scalabilità con diversi numeri di ranks
4. **Summary Report**: Report testuale con statistiche aggregate

## Struttura dei File di Output

```
results/
├── oneccl/                          # Directory risultati CSV
│   ├── oneccl_allreduce_int_results.csv
│   ├── oneccl_allreduce_float_results.csv
│   └── oneccl_allreduce_double_results.csv
└── plots/                           # Directory grafici
    ├── oneccl_allreduce_performance_vs_message_size.png
    ├── oneccl_allreduce_performance_by_datatype.png
    ├── oneccl_allreduce_int_scaling_analysis.png
    └── summary_report.txt
```

## Esempio di Workflow Completo

```bash
# 1. Compila il programma
icpx -o allreduce allreduce.cpp -lccl -lmpi -fsycl

# 2. Esegui i benchmark
./run_benchmark.sh

# 3. Genera i grafici
python3 ../../../scripts/generate_plots.py \
    --input ../../../results/oneccl \
    --output ../../../results/plots

# 4. Visualizza i risultati
ls ../../../results/plots/
cat ../../../results/plots/summary_report.txt
```

## Logging Avanzato

### Console Output
```
[LOG] oneccl allreduce int size=1024 rank=0 time=5.234ms -> ../results/oneccl/oneccl_allreduce_int_results.csv
Rank 0 allreduce time: 5.234 ms
PASSED
```

### File CSV
I file CSV permettono analisi dettagliate con strumenti esterni come:
- **Python/Pandas**: Per analisi statistiche
- **Excel**: Per pivot tables e grafici
- **R**: Per analisi statistiche avanzate
- **Jupyter Notebooks**: Per analisi interattive

## Troubleshooting

### Problemi Comuni

1. **Directory non creata**: Il logger crea automaticamente le directory se non esistono
2. **Permessi di scrittura**: Assicurati di avere permessi di scrittura nella directory di output
3. **File CSV corrotti**: Ogni sessione appende ai file esistenti, mantieni copie di backup
4. **Memoria insufficiente**: Riduci `--count` per messaggi più piccoli

### Debug

Aggiungi debug temporaneo:
```cpp
std::cout << "Debug: logging to " << output_dir << std::endl;
```

## Estensioni Future

Il sistema di logging è progettato per essere estensibile:
- Aggiunta di nuove metriche (bandwidth, latency)
- Supporto per altri formati di output (JSON, HDF5)
- Integrazione con sistemi di monitoring (Prometheus, InfluxDB)
- Analisi automatiche (anomaly detection, regression analysis)
