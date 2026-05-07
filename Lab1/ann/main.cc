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
#include "pq_quantization.h"

using namespace std;

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
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
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

struct SearchResult
{
    float recall;
    int64_t latency;
};

void run_evaluation(int thread_count, int top_c, PQQuantizer& pq_engine, 
                    const PQCode* aligned_base_codes, const float* base, 
                    const float* test_query, const int* test_gt, 
                    size_t test_number, size_t base_number, size_t vecdim, size_t test_gt_d, size_t k,
                    std::ofstream& csv_file) {
    
    omp_set_num_threads(thread_count);
    MicroProfiler::reset(); 

    std::vector<SearchResult> results(test_number);

    for(size_t i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        auto res = pq_engine.search(aligned_base_codes, base, test_query + i * vecdim, base_number, k, top_c);

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
            int x = res.top().second;
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

    std::cerr << "[Result] Threads: " << thread_count << " | Top_C: " << top_c 
              << " | Recall: " << final_recall << " | Latency: " << final_latency << " us\n";
              
    // 将宏观结果写入 CSV
    if (csv_file.is_open()) {
        csv_file << thread_count << "," << top_c << "," << final_recall << "," << final_latency << "\n";
    }
    
    // 打印并保存微观阶段耗时
    std::string profiler_log_path = "files/profiler_detail_T" + std::to_string(thread_count) + "_C" + std::to_string(top_c) + ".csv";
    MicroProfiler::print_and_save(profiler_log_path);
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    const size_t k = 10;

    // 离线阶段：训练与编码
    PQQuantizer pq_engine;
    pq_engine.train(base, base_number);

    PQCode* aligned_base_codes = nullptr;
    if (posix_memalign((void**)&aligned_base_codes, 64, base_number * sizeof(PQCode)) != 0) {
        std::cerr << "Memory alignment allocation failed!" << "\n";
        return -1;
    }

    pq_engine.encode_batch(base, aligned_base_codes, base_number);

    std::ofstream csv_file("files/latency_recall_tradeoff.csv", std::ios::app);
    if (csv_file.is_open()) {
        csv_file << "Threads,Top_C,Recall@10,Latency(us)\n";
    }

    // 参数扫描空间设计
    // 1. 测试不同线程加速比
    std::vector<int> thread_configs = {1, 2, 4, 8}; 
    // 2. 测试召回率与延迟的折中，验证 ADC 查表的 Trade-off
    std::vector<int> top_c_configs = {20, 50, 100, 200, 500}; 

    std::cerr << "\n[System] Starting Automated Grid Search Evaluation...\n";

    for (int t : thread_configs) {
        for (int c : top_c_configs) {
            std::cerr << "\n>>> Running config: Threads=" << t << ", Top_C=" << c << "\n";
            run_evaluation(t, c, pq_engine, aligned_base_codes, base, test_query, test_gt, 
                           test_number, base_number, vecdim, test_gt_d, k, csv_file);
        }
    }

    if (csv_file.is_open()) {
        csv_file.close();
    }

    std::cerr << "\n[System] All evaluations completed. Check files/ directory for results.\n";

    free(aligned_base_codes);
    delete[] base; 
    delete[] test_query;
    delete[] test_gt;

    return 0;
}