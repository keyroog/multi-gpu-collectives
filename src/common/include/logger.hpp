#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

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
            std::filesystem::create_directories(output_dir);
        }
    }
    
    bool file_exists(const std::string& filename) const {
        return std::filesystem::exists(filename);
    }
    
    void write_header(std::ofstream& file) const {
        file << "timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_number,time_ms,environment\n";
    }

    std::string capture_environment() const {
        std::stringstream env_info;
        
        // Capture relevant environment variables for debugging
        const char* ccl_log = std::getenv("CCL_LOG_LEVEL");
        const char* nccl_debug = std::getenv("NCCL_DEBUG");
        const char* omp_threads = std::getenv("OMP_NUM_THREADS");
        const char* ze_affinity = std::getenv("ZE_AFFINITY_MASK");
        const char* mpi_ranks = std::getenv("OMPI_COMM_WORLD_SIZE");
        
        env_info << "CCL_LOG=" << (ccl_log ? ccl_log : "none") << ";";
        env_info << "NCCL_DEBUG=" << (nccl_debug ? nccl_debug : "none") << ";";
        env_info << "OMP_THREADS=" << (omp_threads ? omp_threads : "auto") << ";";
        env_info << "ZE_AFFINITY=" << (ze_affinity ? ze_affinity : "none") << ";";
        env_info << "MPI_SIZE=" << (mpi_ranks ? mpi_ranks : "unknown");
        
        return env_info.str();
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
        
        // Capture environment information
        std::string env_info = capture_environment();
        
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
             << std::fixed << std::setprecision(3) << time_ms << ","
             << env_info << "\n";
        
        file.close();
        
        // Log anche su console per debug
        std::cout << "[LOG] " << library_name << " " << collective_name 
                  << " " << data_type << " size=" << message_size_elements 
                  << " rank=" << rank << " run=" << current_run_id
                  << " time=" << time_ms << "ms -> " << filename << std::endl;
    }
    
    void log_summary(const std::string& data_type, size_t message_size_elements, int num_ranks, 
                    double min_time_ms, double max_time_ms, double avg_time_ms) {
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Library: " << library_name << std::endl;
        std::cout << "Collective: " << collective_name << std::endl;
        std::cout << "Data Type: " << data_type << std::endl;
        std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
        std::cout << "Number of Ranks: " << num_ranks << std::endl;
        std::cout << "Current Run: " << current_run_id << std::endl;
        std::cout << "Min Time: " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "Goodput (worst-rank): " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "===============\n" << std::endl;
    }
    
    // Metodo per calcolare goodput (worst-rank timing)
    static double calculate_goodput(const std::vector<double>& rank_times) {
        if (rank_times.empty()) return 0.0;
        return *std::max_element(rank_times.begin(), rank_times.end());
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
        std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, run_number, time_ms, environment" << std::endl;
        std::cout << "\nEnvironment Variables for Performance Analysis:" << std::endl;
        std::cout << "  CCL_LOG_LEVEL=trace     : Enable OneCCL detailed logging" << std::endl;
        std::cout << "  NCCL_DEBUG=TRACE       : Enable NCCL detailed logging" << std::endl;
        std::cout << "  ZE_DEBUG=1             : Enable Level-Zero debug info" << std::endl;
        std::cout << "  ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 : Consistent device ordering" << std::endl;
    }
};

// Definition of static member
int Logger::global_run_counter = 0;
