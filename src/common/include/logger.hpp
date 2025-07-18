#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
extern char **environ;

class Logger {
private:
    std::string output_dir;
    std::string library_name;
    std::string collective_name;
    int run_id;
    int mpi_rank;

    // Nuove variabili per informazioni sui nodi
    std::string hostname;
    int node_id;
    int total_nodes;
    bool is_multi_node;

    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
    
    std::string get_hostname_internal() const {
        char hostname_buf[256];
        if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0) {
            return std::string(hostname_buf);
        }
        return "unknown";
    }
    
    void initialize_node_info() {
        // Ottieni hostname
        hostname = get_hostname_internal();

        // Ottieni informazioni MPI sui nodi
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        mpi_rank = rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Raccogli tutti gli hostname dai vari rank
        const int max_hostname_len = 256;
        char local_hostname[max_hostname_len];
        strncpy(local_hostname, hostname.c_str(), max_hostname_len - 1);
        local_hostname[max_hostname_len - 1] = '\0';
        
        // Buffer per raccogliere tutti gli hostname
        std::vector<char> all_hostnames(size * max_hostname_len);
        
        // Gather di tutti gli hostname al rank 0
        MPI_Allgather(local_hostname, max_hostname_len, MPI_CHAR,
                     all_hostnames.data(), max_hostname_len, MPI_CHAR,
                     MPI_COMM_WORLD);
        
        // Analizza gli hostname per determinare nodi unici
        std::unordered_set<std::string> unique_hosts;
        std::vector<std::string> host_list;
        
        for (int i = 0; i < size; i++) {
            std::string host(all_hostnames.data() + i * max_hostname_len);
            host_list.push_back(host);
            unique_hosts.insert(host);
        }
        
        // Determina total_nodes
        total_nodes = static_cast<int>(unique_hosts.size());
        
        // Determina is_multi_node
        is_multi_node = (total_nodes > 1);
        
        // Determina node_id (assegna un ID progressivo basato sull'ordine degli hostname unici)
        std::vector<std::string> sorted_hosts(unique_hosts.begin(), unique_hosts.end());
        std::sort(sorted_hosts.begin(), sorted_hosts.end());
        
        node_id = 0;
        for (int i = 0; i < sorted_hosts.size(); i++) {
            if (sorted_hosts[i] == hostname) {
                node_id = i;
                break;
            }
        }
        
        // Debug output (solo dal rank 0)
        if (rank == 0) {
            std::cout << "[NODE INFO] Total nodes: " << total_nodes 
                      << ", Multi-node: " << (is_multi_node ? "yes" : "no") << std::endl;
            std::cout << "[NODE INFO] Unique hostnames: ";
            for (const auto& host : sorted_hosts) {
                std::cout << host << " ";
            }
            std::cout << std::endl;
        }
    }

    // std::string capture_env() const {
    //     std::string result;
    //     for (char **env = environ; env && *env; ++env) {
    //         std::string entry(*env);
    //         if (entry.rfind(env_var_prefix, 0) == 0) {
    //             if (!result.empty()) result += ";";
    //             result += entry;
    //         }
    //     }
    //     return result;
    // }
    
    std::string get_filename(const std::string& data_type, size_t message_size_elements) const {
        return output_dir + "/" + library_name + "_" + collective_name + "_" +
               data_type + "_" + std::to_string(message_size_elements) + "_results.csv";
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
            
            // Trova la colonna run_id (posizione 12, 0-indexed - aggiornato per le nuove colonne)
            std::stringstream ss(line);
            std::string token;
            int column = 0;
            
            while (std::getline(ss, token, ',') && column <= 12) {
                if (column == 12) { // Colonna run_id (aggiornata)
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
        file << "timestamp,library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,rank,hostname,node_id,total_nodes,is_multi_node,run_id,time_ms,gdr_copy\n";
    }

public:
    Logger(const std::string& output_dir, const std::string& library_name, const std::string& collective_name)
        : output_dir(output_dir), library_name(library_name), collective_name(collective_name) {
        ensure_directory_exists();
        initialize_node_info();
        // set default prefix based on library
        if (library_name.find("nccl") != std::string::npos || library_name.find("NCCL") != std::string::npos)
            // env_var_prefix = "NCCL_";
            ;
        else if (library_name.find("ccl") != std::string::npos || library_name.find("CCL") != std::string::npos)
            // env_var_prefix = "CCL_";
            ;
        else
            // env_var_prefix = "NCCL_"; // fallback
            ;
        // env_vars = capture_env();
        run_id = get_next_run_id();
    }
    // optionally override which prefix to capture
    void set_env_prefix(const std::string& prefix) {
        // env_var_prefix = prefix;
        // env_vars = capture_env();
    }
    
    void log_result(const std::string& data_type, size_t message_size_elements, int num_ranks, int rank, double time_ms, bool used_gdr_copy = false) {
        // Determine if GDR Copy is enabled via NCCL environment variable
        const char* gdr_env = std::getenv("NCCL_ENABLE_GDRCOPY");
        used_gdr_copy = (gdr_env && std::string(gdr_env) == "1");
        if (output_dir.empty()) {
            // Se non è specificato un output directory, non loggare su file
            return;
        }
        
        std::string filename = get_filename(data_type, message_size_elements);
        
        // Sincronizzazione MPI per gestire la scrittura dell'header
        // Solo il rank 0 controlla e scrive l'header se necessario
        int needs_header = 0;
        if (rank == 0) {
            bool is_new_file = !file_exists(filename);
            needs_header = is_new_file ? 1 : 0;
        }
        
        // Broadcast del flag header a tutti i rank
        MPI_Bcast(&needs_header, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Solo il rank 0 scrive l'header se necessario
        if (needs_header && rank == 0) {
            std::ofstream header_file(filename, std::ios::app);
            if (header_file.is_open()) {
                write_header(header_file);
                header_file.close();
            }
        }
        
        // Sincronizzazione per assicurare che l'header sia scritto prima dei dati
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Ora tutti i rank possono scrivere i loro dati
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open log file: " << filename << std::endl;
            return;
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
             << hostname << ","
             << node_id << ","
             << total_nodes << ","
             << (is_multi_node ? "true" : "false") << ","
             << run_id << ","
             << std::fixed << std::setprecision(3) << time_ms << ","
             << (used_gdr_copy ? "true" : "false") << "\n";
        
        file.close();
        
        // Log anche su console per debug
        std::cout << "[LOG] " << library_name << " " << collective_name 
                  << " " << data_type << " size=" << message_size_elements 
                  << " rank=" << rank << " hostname=" << hostname << " node=" << node_id
                  << " run=" << run_id << " time=" << time_ms << "ms" << " -> " << filename << std::endl;
    }
    
    void log_summary(const std::string& data_type, size_t message_size_elements, int num_ranks, 
                     double min_time_ms, double max_time_ms, double avg_time_ms) {
         std::cout << "\n=== SUMMARY ===" << std::endl;
         std::cout << "Library: " << library_name << std::endl;
         std::cout << "Collective: " << collective_name << std::endl;
         std::cout << "Data Type: " << data_type << std::endl;
         std::cout << "Message Size: " << message_size_elements << " elements" << std::endl;
         std::cout << "Number of Ranks: " << num_ranks << std::endl;
         std::cout << "Hostname: " << hostname << std::endl;
         std::cout << "Node ID: " << node_id << std::endl;
         std::cout << "Total Nodes: " << total_nodes << std::endl;
         std::cout << "Multi-Node: " << (is_multi_node ? "Yes" : "No") << std::endl;
         std::cout << "Run ID: " << run_id << std::endl;
         // std::cout << "Env Vars: " << env_vars << std::endl;
         std::cout << "Min Time: " << std::fixed << std::setprecision(3) << min_time_ms << " ms" << std::endl;
         std::cout << "Max Time: " << std::fixed << std::setprecision(3) << max_time_ms << " ms" << std::endl;
         std::cout << "Avg Time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "===============\n" << std::endl;
    }
    
    // Metodo per impostare manualmente il run_id (opzionale)
    void set_run_id(int new_run_id) {
        run_id = new_run_id;
    }
    
    // Metodi getter per le nuove informazioni sui nodi
    const std::string& get_hostname() const { return hostname; }
    int get_node_id() const { return node_id; }
    int get_total_nodes() const { return total_nodes; }
    bool get_is_multi_node() const { return is_multi_node; }
    
    // Metodo per ottenere il run_id corrente
    int get_run_id() const {
        return run_id;
    }

    bool check_if_gdr_copy_used(const std::string& trace_log_file) const {
        std::ifstream file(trace_log_file);
        if (!file.is_open()) {
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            // Cerca frasi indicative
            if (line.find("GDRCopy") != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    static void print_usage() {
        std::cout << "\nLogger Usage:" << std::endl;
        std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
        std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
        std::cout << "\nOutput format: CSV files with columns:" << std::endl;
        std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, time_ms, gdr_copy" << std::endl;
    }

    void log_result_with_gdr_detection(const std::string& data_type, 
                                            size_t message_size_elements, 
                                            int num_ranks, 
                                            int rank, 
                                            double time_ms) {
        std::string trace_log_file = output_dir + "/nccl_" + collective_name 
                                    + "_trace." + hostname 
                                    + "." + std::to_string(getpid()) + ".log";

        bool used_gdr_copy = check_if_gdr_copy_used(trace_log_file);
        log_result(data_type, message_size_elements, num_ranks, rank, time_ms, used_gdr_copy);
    }
};