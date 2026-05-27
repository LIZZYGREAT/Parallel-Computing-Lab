#pragma once
#include <chrono>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>

template<typename T = void>
class MicroProfilerT {
public:
    static std::map<std::string, double> times;
    
    struct Timer {
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        
        Timer(std::string n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed = end - start;
            // 确保多线程环境下的数据竞争安全
            #pragma omp critical(micro_profiler)
            {
                MicroProfilerT::times[name] += elapsed.count();
            }
        }
    };

    static void print_and_save(const std::string& filepath, size_t num_samples = 1) {
        std::ofstream fout(filepath);
        std::cerr << "========== Micro-Profiling Results ==========\n";
        if (num_samples > 1) {
            std::cerr << "Samples: " << num_samples << " (per-query avg shown)\n";
        }

        double total = 0.0;
        for (const auto& pair : times) {
            total += pair.second;
        }

        for (const auto& pair : times) {
            double pct = total > 0.0 ? (pair.second / total * 100.0) : 0.0;
            double avg = pair.second / static_cast<double>(num_samples);
            std::cerr << pair.first << ": " << pair.second << " us total, "
                      << avg << " us/avg (" << pct << "%)\n";
            if (fout.is_open()) {
                fout << pair.first << "," << pair.second << "," << avg << "\n";
            }
        }
        std::cerr << "Total profiled: " << total << " us";
        if (num_samples > 1) {
            std::cerr << " (" << total / num_samples << " us/query avg)";
        }
        std::cerr << "\n=============================================\n";
        if (fout.is_open()) fout.close();
    }

    static void reset() { 
        times.clear(); 
    }
};

// 静态成员初始化
template<typename T>
std::map<std::string, double> MicroProfilerT<T>::times;

using MicroProfiler = MicroProfilerT<>;