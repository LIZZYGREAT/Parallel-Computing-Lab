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
            #pragma omp critical
            {
                MicroProfilerT::times[name] += elapsed.count();
            }
        }
    };

    static void print_and_save(const std::string& filepath) {
        std::ofstream fout(filepath, std::ios::app);
        std::cerr << "========== Micro-Profiling Results ==========\n";
        for (const auto& pair : times) {
            std::cerr << pair.first << ": " << pair.second << " us\n";
            if(fout.is_open()) {
                fout << pair.first << "," << pair.second << "\n";
            }
        }
        std::cerr << "=============================================\n";
        if(fout.is_open()) fout.close();
    }

    static void reset() { 
        times.clear(); 
    }
};

// 静态成员初始化
template<typename T>
std::map<std::string, double> MicroProfilerT<T>::times;

using MicroProfiler = MicroProfilerT<>;