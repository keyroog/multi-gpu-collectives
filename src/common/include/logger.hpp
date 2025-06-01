#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

class Logger {
private:
    std::string output_dir;
    std::string library_name;
    std::string collective_name;
    int run_id;
    
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
    
    int get_next_run_id() const {
        if (output_dir.empty()) {
            return 1; // Se non c'è output directory, usa sempre run_id = 1
        }
        
        // Cerca tutti i file CSV esistenti per determinare il prossimo run_id
        int max_run_id = 0;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                    std::string filename = entry.path().filename().string();
                    // Controlla se il file appartiene a questo logger (stesso library e collective)
                    std::string expected_prefix = library_name + "_" + collective_name + "_";
                    if (filename.find(expected_prefix) == 0) {
                        // Leggi il file per trovare il run_id massimo
                        int file_max_run = get_max_run_id_from_file(entry.path().string());
                        max_run_id = std::max(max_run_id, file_max_run);
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error&) {
            // Se c'è un errore nell'accesso alla directory, usa run_id = 1
            return 1;
        }
        
        return max_run_id + 1;
    }
    
    int get_max_run_id_from_file(const std::string& filename) const {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return 0;
        }
        
        std::string line;
        int max_run_id = 0;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Salta l'header
            }
            
            // Trova la colonna run_id (posizione 8, 0-indexed)
            std::stringstream ss(line);
            std::string token;
            int column = 0;
            
            while (std::getline(ss, token, ',') && column <= 8) {
                if (column == 8) { // Colonna run_id
                    try {
                        int run_id = std::stoi(token);
                        max_run_id = std::max(max_run_id, run_id);
                    } catch (const std::exception&) {
                        // Ignora righe malformate
                    }
                    break;
                }
                column++;
            }
        }
        
        return max_run_id;
    }
    
    void write_header(std::ofstream& file) const {
        file << "timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_id,time_ms\n";
    }

public:
    Logger(const std::string& output_dir, const std::string& library_name, const std::string& collective_name)
        : output_dir(output_dir), library_name(library_name), collective_name(collective_name) {
        ensure_directory_exists();
        run_id = get_next_run_id();
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
             << run_id << ","
             << std::fixed << std::setprecision(3) << time_ms << "\n";
        
        file.close();
        
        // Log anche su console per debug
        std::cout << "[LOG] " << library_name << " " << collective_name 
                  << " " << data_type << " size=" << message_size_elements 
                  << " rank=" << rank << " run=" << run_id << " time=" << time_ms << "ms -> " << filename << std::endl;
    }
    
    void log_summary(const std::string& data_type, size_t message_size_elements, int num_ranks, 
                    double min_time_ms, double max_time_ms, double avg_time_ms) {
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Library: " << library_name << std::endl;
        std::cout << "Collective: " << collective_name << std::endl;
        std::cout << "Data Type: " << data_type << std::endl;
        std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
        std::cout << "Number of Ranks: " << num_ranks << std::endl;
        std::cout << "Run ID: " << run_id << std::endl;
        std::cout << "Min Time: " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "===============\n" << std::endl;
    }
    
    // Metodo per impostare manualmente il run_id (opzionale)
    void set_run_id(int new_run_id) {
        run_id = new_run_id;
    }
    
    // Metodo per ottenere il run_id corrente
    int get_run_id() const {
        return run_id;
    }
    
    static void print_usage() {
        std::cout << "\nLogger Usage:" << std::endl;
        std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
        std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
        std::cout << "\nOutput format: CSV files with columns:" << std::endl;
        std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, run_id, time_ms" << std::endl;
    }
};