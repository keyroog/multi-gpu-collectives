#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <stdexcept>

class Logger {
public:
    static void append(const std::string& path, const std::vector<std::string>& fields) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        std::ofstream fout(path, std::ios::app);
        if (!fout)
            throw std::runtime_error("Cannot open log file: " + path);
        for (size_t i = 0; i < fields.size(); ++i) {
            fout << fields[i] << (i + 1 < fields.size() ? ',' : '\n');
        }
    }
};