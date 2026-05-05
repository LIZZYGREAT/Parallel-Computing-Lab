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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "scan.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
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

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 1. 使用 posix_memalign 分配 32 字节对齐的内存
    float* aligned_aosoa_base = nullptr;
    if (posix_memalign((void**)&aligned_aosoa_base, 32, base_number * vecdim * sizeof(float)) != 0) {
        std::cerr << "Memory alignment allocation failed!" << "\n";
        return -1;
    }

    // 2. 将原生的 AoS 数据重排为 AoSoA (Block Size = 4)
    const size_t simd_width = 4;
    for (size_t block_idx = 0; block_idx < base_number / simd_width; ++block_idx) {
        size_t block_start_aligned = block_idx * simd_width * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            for (size_t v = 0; v < simd_width; ++v) {
                float val = base[(block_idx * simd_width + v) * vecdim + d];
                aligned_aosoa_base[block_start_aligned + d * simd_width + v] = val;
            }
        }
    }
    
    // 处理可能的尾部数据
    size_t tail_start = (base_number / simd_width) * simd_width;
    for (size_t i = tail_start; i < base_number; ++i) {
        for (size_t d = 0; d < vecdim; ++d) {
            aligned_aosoa_base[tail_start * vecdim + (i - tail_start) * vecdim + d] = base[i * vecdim + d];
        }
    }

    // 3. 预先分配用于搜索阶段的全局结果缓冲
    QueryResult* global_results = new QueryResult[base_number];
    // =========================================================================

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        auto res = flat_search_aosoa(aligned_aosoa_base, test_query + i*vecdim, base_number, vecdim, k, global_results);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    
    // 释放内存
    free(aligned_aosoa_base);
    delete[] global_results;
    delete[] base; 
    delete[] test_query;
    delete[] test_gt;

    return 0;
}