#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <stdlib.h> 
#include <omp.h>
#include <queue>
#include <ctime>

#include "profiler.h"
#include "ivfpq_index.h"
#include "adc_searcher.h"
#include "sdc_searcher.h"
#include "opq_rotate.h"

using namespace std;

// 异步持久化日志记录
void save_persistent_log(float avg_recall, float avg_latency) {
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::localtime(&now);

    std::stringstream ss;
    ss << "files/result_" 
       << (1900 + ltm->tm_year) << std::setw(2) << std::setfill('0') << (1 + ltm->tm_mon) 
       << std::setw(2) << std::setfill('0') << ltm->tm_mday << "_" 
       << std::setw(2) << std::setfill('0') << ltm->tm_hour 
       << std::setw(2) << std::setfill('0') << ltm->tm_min << ".log";

    std::ofstream fout(ss.str());
    if (fout.is_open()) {
        fout << "Average Recall: " << avg_recall << std::endl;
        fout << "Average Latency: " << avg_latency << " us" << std::endl;
        fout << "Timestamp: " << std::asctime(ltm);
        fout.close();
        std::cerr << "[System] Result successfully saved to " << ss.str() << std::endl;
    } else {
        std::cerr << "[Error] Failed to write to files/ directory! Please check permissions." << std::endl;
    }
}

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[Error] Cannot open file " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult {
    float recall;
    int64_t latency;
};

// 评估逻辑：接收 BaseSearcher 接口，解除具体算法强耦合
void run_evaluation(int thread_count, int nprobe, BaseSearcher* searcher, 
                    const float* test_query, const int* test_gt, 
                    size_t test_number, size_t vecdim, size_t test_gt_d, size_t k,
                    std::ofstream& csv_file, const std::string& method_name, bool use_opq) {
    
    omp_set_num_threads(thread_count);
    MicroProfiler::reset(); 

    std::vector<SearchResult> results(test_number);
    std::vector<float> query_buf(vecdim);

    for(size_t i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        const float* q = test_query + i * vecdim;
        if (use_opq) {
            opq_rotate(q, query_buf.data(), static_cast<int>(vecdim));
            q = query_buf.data();
        }
        auto res = searcher->search(q, k, nprobe);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (!res.empty()) {   
            uint32_t x = res.top().id;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    float final_recall = avg_recall / test_number;
    float final_latency = avg_latency / test_number;

    std::cerr << "[Result] Method: " << method_name << " | Threads: " << thread_count 
              << " | NProbe: " << nprobe << " | Recall: " << final_recall 
              << " | Latency: " << final_latency << " us\n";

    std::stringstream profiler_path;
    profiler_path << "files/profiler_detail_" << method_name
                  << "_T" << thread_count << "_P" << nprobe << ".csv";
    MicroProfiler::print_and_save(profiler_path.str(), test_number);
              
    if (csv_file.is_open()) {
        csv_file << method_name << "," << thread_count << "," << nprobe << "," 
                 << final_recall << "," << final_latency << "\n";
    }
}

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    

    // 测试样本量与 Top-K 设定
    test_number = 20;
    const size_t k = 10;

    int n_lists = 1024;
    int M = 16;
    const bool use_opq = false; 
    std::vector<float> rotated_base;
    const float* base_for_build = base;
    
    if (use_opq) {
        rotated_base.resize(base_number * vecdim);
        opq_rotate_batch(base, rotated_base.data(), base_number, static_cast<int>(vecdim));
        base_for_build = rotated_base.data(); 
    }

    IVFPQIndex index(vecdim, n_lists);
    
    MicroProfiler::Timer _t_build("Index_Build");
    index.build(base_for_build, base_number);
    MicroProfiler::print_and_save("files/profiler_build.csv");

    MicroProfiler::reset();
    
    ADCSearcher adc_searcher(&index, base_for_build, 30);
    MicroProfiler::print_and_save("files/profiler_adc_init.csv");
    
    MicroProfiler::reset();
    SDCSearcher sdc_searcher(&index, base_for_build, 30); 
    MicroProfiler::print_and_save("files/profiler_sdc_init.csv");

    const int profile_threads = 4;
    const int profile_nprobe = 64;

    std::ofstream csv_file("files/ivfpq_query_profile.csv", std::ios::trunc);
    if (csv_file.is_open()) {
        csv_file << "Method,Threads,NProbe,Recall@10,Latency(us)\n";
    }

    std::cerr << "\n[System] Query profiling: Threads=" << profile_threads
              << ", NProbe=" << profile_nprobe << "\n";

    std::cerr << "\n>>> ADC\n";
    run_evaluation(profile_threads, profile_nprobe, &adc_searcher, test_query, test_gt,
                   test_number, vecdim, test_gt_d, k, csv_file, "ADC", use_opq);

    std::cerr << "\n>>> SDC\n";
    run_evaluation(profile_threads, profile_nprobe, &sdc_searcher, test_query, test_gt,
                   test_number, vecdim, test_gt_d, k, csv_file, "SDC", use_opq);

    if (csv_file.is_open()) {
        csv_file.close();
    }

    std::cerr << "\n[System] Done. Profiler: files/profiler_detail_*_T"
              << profile_threads << "_P" << profile_nprobe << ".csv\n";
    std::cerr << "[System] Plot: python3 viz/run_query_profile.py\n";

    delete[] base; 
    delete[] test_query;
    delete[] test_gt;

    return 0;
}