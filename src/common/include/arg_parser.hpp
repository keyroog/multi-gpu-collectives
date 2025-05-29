#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

class ArgParser {
public:
    ArgParser(int argc, char** argv) {
        for (int i = 1; i + 1 < argc; i += 2) {
            args_[argv[i]] = argv[i + 1];
        }
    }

    template<class T>
    ArgParser& add(const std::string& key) {
        required_.push_back(key);
        return *this;
    }

    void parse() {
        for (auto& key : required_) {
            if (!args_.count(key))
                throw std::runtime_error("Missing argument: " + key);
        }
    }

    template<class T>
    T get(const std::string& key) const {
        auto it = args_.find(key);
        if (it == args_.end())
            throw std::runtime_error("Argument not found: " + key);
        std::istringstream ss(it->second);
        T val;
        if (!(ss >> val))
            throw std::runtime_error("Invalid value for " + key);
        return val;
    }

private:
    std::unordered_map<std::string, std::string> args_;
    std::vector<std::string> required_;
};