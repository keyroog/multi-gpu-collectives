#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <unistd.h>        // per gethostname
#include <vector>
#include <algorithm>
#include <map>

// Forward declarations per evitare include di MPI header
extern "C" {
    int MPI_Allgather(const void *sendbuf, int sendcount, int sendtype,
                     void *recvbuf, int recvcount, int recvtype, int comm);
}

class Logger {
private:
    std::string output_dir;
    std::string library_name;
    std::string collective_name;
    int run_id;
    std::string hostname;
    int node_id;
    int total_nodes;
    bool is_multi_node;
    
    std::string get_hostname() const {
        char hostname_buffer[256];
        if (gethostname(hostname_buffer, sizeof(hostname_buffer)) == 0) {
            return std::string(hostname_buffer);
        }
        return "unknown";
    }
    
    std::pair<int, int> get_node_info(int rank, int num_ranks) const {
        // Raccogli hostname da tutti i rank
        char local_hostname[256];
        gethostname(local_hostname, sizeof(local_hostname));
        
        // Crea buffer per ricevere tutti gli hostname
        std::vector<char> all_hostnames(num_ranks * 256);
        
        // Gather di tutti gli hostname usando la definizione locale di MPI_CHAR
        const int MPI_CHAR = 1;
        const int MPI_COMM_WORLD = 0;
        MPI_Allgather(local_hostname, 256, MPI_CHAR, 
                      all_hostnames.data(), 256, MPI_CHAR, MPI_COMM_WORLD);
        
        // Analizza gli hostname per determinare nodi unici
        std::map<std::string, int> hostname_to_node_id;
        std::vector<std::string> unique_hostnames;
        
        for (int i = 0; i < num_ranks; i++) {
            std::string hostname_str(&all_hostnames[i * 256]);
            // Rimuovi caratteri null e spazi
            hostname_str = hostname_str.substr(0, hostname_str.find('\0'));
            
            if (hostname_to_node_id.find(hostname_str) == hostname_to_node_id.end()) {
                hostname_to_node_id[hostname_str] = unique_hostnames.size();
                unique_hostnames.push_back(hostname_str);
            }
        }
        
        // Trova il node_id per questo rank
        std::string my_hostname(local_hostname);
        my_hostname = my_hostname.substr(0, my_hostname.find('\0'));
        int my_node_id = hostname_to_node_id[my_hostname];
        int total_nodes = unique_hostnames.size();
        
        return std::make_pair(my_node_id, total_nodes);
    }
    
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
            
            // Trova la colonna run_id (ora posizione 8, 0-indexed)
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
        file << "timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,run_id,hostname,node_id,total_nodes,is_multi_node,time_ms\n";
    }

public:
    Logger(const std::string& output_dir, const std::string& library_name, const std::string& collective_name, 
           int rank = 0, int num_ranks = 1)
        : output_dir(output_dir), library_name(library_name), collective_name(collective_name) {
        ensure_directory_exists();
        run_id = get_next_run_id();
        hostname = get_hostname();
        
        // Ottieni informazioni sui nodi solo se MPI è inizializzato
        auto node_info = get_node_info(rank, num_ranks);
        node_id = node_info.first;
        total_nodes = node_info.second;
        is_multi_node = (total_nodes > 1);
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
             << hostname << ","
             << node_id << ","
             << total_nodes << ","
             << (is_multi_node ? "true" : "false") << ","
             << std::fixed << std::setprecision(3) << time_ms << "\n";
        
        file.close();
        
        // Log anche su console per debug
        std::cout << "[LOG] " << library_name << " " << collective_name 
                  << " " << data_type << " size=" << message_size_elements 
                  << " rank=" << rank << " node=" << hostname << "(" << node_id << ")"
                  << " run=" << run_id << " time=" << time_ms << "ms"
                  << " [" << (is_multi_node ? "MULTI-NODE" : "SINGLE-NODE") << "]"
                  << " -> " << filename << std::endl;
    }
    
    void log_summary(const std::string& data_type, size_t message_size_elements, int num_ranks, 
                    double min_time_ms, double max_time_ms, double avg_time_ms) {
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Library: " << library_name << std::endl;
        std::cout << "Collective: " << collective_name << std::endl;
        std::cout << "Data Type: " << data_type << std::endl;
        std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
        std::cout << "Number of Ranks: " << num_ranks << std::endl;
        std::cout << "Total Nodes: " << total_nodes << std::endl;
        std::cout << "Execution Type: " << (is_multi_node ? "MULTI-NODE" : "SINGLE-NODE") << std::endl;
        std::cout << "Run ID: " << run_id << std::endl;
        std::cout << "Min Time: " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
        std::cout << "Max Time: " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
        std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "===============\n" << std::endl;
    }
    
    }
    
    // Metodo per impostare manualmente il run_id (opzionale)
    void set_run_id(int new_run_id) {
        run_id = new_run_id;
    }
    
    // Metodo per ottenere informazioni sui nodi
    bool get_is_multi_node() const { return is_multi_node; }
    int get_node_id() const { return node_id; }
    int get_total_nodes() const { return total_nodes; }
    std::string get_hostname_info() const { return hostname; }
    int get_run_id() const { return run_id; }
    
    static void print_usage() {
        std::cout << "\nLogger Usage:" << std::endl;
        std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
        std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
        std::cout << "\nOutput format: CSV files with columns:" << std::endl;
        std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, run_id, hostname, node_id, total_nodes, is_multi_node, time_ms" << std::endl;
    }
};