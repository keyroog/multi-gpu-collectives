#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <sys/stat.h>

class Logger {
private:
    std::string output_dir;
    std::string library_name;
    std::string collective_name;
    static int global_run_counter;  // Static counter for complete benchmark runs
    int current_run_id;
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
    
    std::string get_filename(const std::string& data_type) const {
        return output_dir + "/" + library_name + "_" + collective_name + "_" + data_type + "_results.csv";
    }
    
    void ensure_directory_exists() const {
        if (!output_dir.empty()) {
            // Simple directory creation using system command for compatibility
            std::string cmd = "mkdir -p \"" + output_dir + "\"";
            std::system(cmd.c_str());
        }
    }
    
    bool file_exists(const std::string& filename) const {
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }
    
    void write_header(std::ofstream& file) const {
        file << "timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms\n";
    }

public:
    Logger(const std::string& output_dir, const std::string& library_name, const std::string& collective_name)
        : output_dir(output_dir), library_name(library_name), collective_name(collective_name) {
        ensure_directory_exists();
        current_run_id = global_run_counter;  // Capture current run ID
    }
    
    // Static method to start a new benchmark run (call before each complete benchmark execution)
    static void start_new_run() {
        global_run_counter++;
    }
    
    // Get current run ID
    static int get_current_run_id() {
        return global_run_counter;
    }
    
    void log_result(const std::string& data_type, size_t message_size_elements, int num_ranks, int rank, double time_ms) {
        if (output_dir.empty()) {
            // Se non è specificato un output directory, non loggare su file
            return;
        }
        
        std::string filename = get_filename(data_type);
        bool is_new_file = !file_exists(filename);
        
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open log file: " << filename << std::endl;
            return;
        }
        
        // Scrivi header se è un nuovo file
        if (is_new_file) {
            write_header(file);
        }
        
        // Calcola la dimensione in bytes (approssimativa)
        size_t element_size = 0;
        if (data_type == "int") element_size = sizeof(int);
        else if (data_type == "float") element_size = sizeof(float);
        else if (data_type == "double") element_size = sizeof(double);
        
        size_t message_size_bytes = message_size_elements * element_size;
        
        // Scrivi la riga di dati
        file << get_timestamp() << ","
             << library_name << ","
             << collective_name << ","
             << data_type << ","
             << message_size_bytes << ","
             << message_size_elements << ","
             << num_ranks << ","
             << rank << ","
             << current_run_id << ","
             << std::fixed << std::setprecision(3) << time_ms << "\n";
        
        file.close();
        
        // Log anche su console per debug
        std::cout << "[LOG] " << library_name << " " << collective_name 
                  << " " << data_type << " size=" << message_size_elements 
                  << " rank=" << rank << " run=" << current_run_id
                  << " time=" << time_ms << "ms -> " << filename << std::endl;
    }
    
    void log_summary(const std::string& data_type, size_t message_size_elements, int num_ranks, 
                    double min_time_ms, double max_time_ms, double avg_time_ms) {
        std::cout << "\n=== SUMMARY (Run " << current_run_id << ") ===" << std::endl;
        std::cout << "Library: " << library_name << std::endl;
        std::cout << "Collective: " << collective_name << std::endl;
        std::cout << "Data Type: " << data_type << std::endl;
        std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
        std::cout << "Number of Ranks: " << num_ranks << std::endl;
        std::cout << "Min Time: " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "Goodput (worst-rank): " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "===============\n" << std::endl;
    }
    
    // Metodo migliorato per calcolare goodput e statistiche per scenari multi-nodo
    struct CollectiveStats {
        double min_time_ms;
        double max_time_ms;
        double avg_time_ms;
        double goodput_ms;  // tempo del peggior rank
        int num_ranks;
        std::vector<double> rank_times;
    };
    
    static CollectiveStats calculate_collective_stats(const std::vector<double>& rank_times) {
        CollectiveStats stats;
        stats.rank_times = rank_times;
        stats.num_ranks = rank_times.size();
        
        if (rank_times.empty()) {
            stats.min_time_ms = stats.max_time_ms = stats.avg_time_ms = stats.goodput_ms = 0.0;
            return stats;
        }
        
        stats.min_time_ms = *std::min_element(rank_times.begin(), rank_times.end());
        stats.max_time_ms = *std::max_element(rank_times.begin(), rank_times.end());
        stats.goodput_ms = stats.max_time_ms;  // Il goodput è il tempo del peggior rank
        
        double sum = 0.0;
        for (double time : rank_times) {
            sum += time;
        }
        stats.avg_time_ms = sum / rank_times.size();
        
        return stats;
    }
    
    // Metodo per loggare le statistiche complete
    void log_collective_stats(const std::string& data_type, size_t message_size_elements, 
                             const CollectiveStats& stats, const std::string& context = "") {
        std::string prefix = context.empty() ? "" : "[" + context + "] ";
        
        std::cout << "\n=== " << prefix << "COLLECTIVE STATISTICS (Run " << current_run_id << ") ===" << std::endl;
        std::cout << "Library: " << library_name << std::endl;
        std::cout << "Collective: " << collective_name << std::endl;
        std::cout << "Data Type: " << data_type << std::endl;
        std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
        std::cout << "Number of Ranks: " << stats.num_ranks << std::endl;
        std::cout << "Min Time: " << std::fixed << std::setprecision(3) << stats.min_time_ms << " ms" << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(3) << stats.max_time_ms << " ms" << std::endl;
        std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << stats.avg_time_ms << " ms" << std::endl;
        std::cout << "Goodput (worst-rank): " << std::fixed << std::setprecision(3) << stats.goodput_ms << " ms" << std::endl;
        
        // Calcola e mostra la varianza per identificare problemi di bilanciamento
        if (stats.num_ranks > 1) {
            double variance = 0.0;
            for (double time : stats.rank_times) {
                variance += (time - stats.avg_time_ms) * (time - stats.avg_time_ms);
            }
            variance /= stats.num_ranks;
            double std_dev = std::sqrt(variance);
            
            std::cout << "Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev << " ms" << std::endl;
            std::cout << "Coefficient of Variation: " << std::fixed << std::setprecision(2) 
                      << (std_dev / stats.avg_time_ms * 100.0) << "%" << std::endl;
        }
        
        std::cout << "===============\n" << std::endl;
    }
    
    // Metodo per logging delle informazioni GPU/topology
    void log_gpu_topology_info(int rank) const {
        std::cout << "\n=== GPU TOPOLOGY INFO (Rank " << rank << ") ===" << std::endl;
        
        // Log Intel GPU Max information if available
        const char* ze_debug = std::getenv("ZE_DEBUG");
        const char* ze_enable_pci_id_device_order = std::getenv("ZE_ENABLE_PCI_ID_DEVICE_ORDER");
        
        std::cout << "ZE_DEBUG: " << (ze_debug ? ze_debug : "not set") << std::endl;
        std::cout << "ZE_ENABLE_PCI_ID_DEVICE_ORDER: " << (ze_enable_pci_id_device_order ? ze_enable_pci_id_device_order : "not set") << std::endl;
        
        // Note: This is a placeholder for more detailed GPU topology analysis
        // In a real implementation, we would use Level-Zero or SYCL APIs to query:
        // - GPU Max modules and tiles
        // - Xe Link connectivity between GPUs
        // - NUMA topology
        // - Memory bandwidth capabilities
        std::cout << "Note: For detailed GPU topology analysis, use 'ocloc query' or 'level_zero_info' tools" << std::endl;
        std::cout << "Recommendation: Check if GPUs are on same 'MACRO GPU' with Xe Link connectivity" << std::endl;
        std::cout << "=================================\n" << std::endl;
    }
    
    // Metodo per resettare il contatore delle esecuzioni globale
    static void reset_global_run_counter() {
        global_run_counter = 0;
    }
    
    // Metodo per ottenere il numero di esecuzioni corrente
    static int get_global_run_count() {
        return global_run_counter;
    }
    
    static void print_usage() {
        std::cout << "\nLogger Usage:" << std::endl;
        std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
        std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
        std::cout << "\nOutput format: CSV files with columns:" << std::endl;
        std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, run_number, time_ms" << std::endl;
        std::cout << "\nRun Management:" << std::endl;
        std::cout << "  - Each complete benchmark execution gets a unique run_number" << std::endl;
        std::cout << "  - Use Logger::start_new_run() before each new benchmark execution" << std::endl;
        std::cout << "  - Goodput is calculated as the worst-rank timing (maximum time across all ranks)" << std::endl;
        std::cout << "  - Statistics include variance analysis for multi-node performance debugging" << std::endl;
        std::cout << "\nCollective Performance Analysis:" << std::endl;
        std::cout << "  - Single-node: All ranks on same node, optimal Xe Link connectivity" << std::endl;
        std::cout << "  - Multi-node: Ranks distributed across nodes, network communication overhead" << std::endl;
        std::cout << "  - Goodput represents synchronization bottleneck (slowest rank determines overall performance)" << std::endl;
    }
};

// Definition of static member
int Logger::global_run_counter = 0;
