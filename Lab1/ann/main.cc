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

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "scan.h"
#include "sq_quantization.h"

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


inline std::priority_queue<std::pair<float, int>> sq_flat_search(
    const uint8_t* sq_base,
    const uint8_t* sq_query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    QueryResult* global_results
) {
    for (size_t i = 0; i < base_number; ++i) {
        uint32_t dist = compute_L2_distance_sq_neon(sq_query, sq_base + i * vecdim, vecdim);  

        global_results[i].dist = static_cast<float>(dist);
        global_results[i].id = i;
    }

    std::nth_element(global_results, global_results + k, global_results + base_number);

    std::priority_queue<std::pair<float, int>> pq;
    for (size_t i = 0; i < k; ++i) {
        pq.push({global_results[i].dist, (int)global_results[i].id});
    }

    return pq;
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

    std::vector<SearchResult> results;
    results.resize(test_number);


    SQQuantizer sq;
    sq.train(base, base_number, vecdim);

    uint8_t* aligned_sq_base = nullptr;
    if (posix_memalign((void**)&aligned_sq_base, 32, base_number * vecdim * sizeof(uint8_t)) != 0) {
        std::cerr << "Memory alignment allocation failed!" << "\n";
        return -1;
    }

    sq.encode_batch(base, aligned_sq_base, base_number, vecdim);

    uint8_t* sq_query_buf = nullptr;
    if (posix_memalign((void**)&sq_query_buf, 32, vecdim * sizeof(uint8_t)) != 0) {
        std::cerr << "Query buffer alignment allocation failed!" << "\n";
        return -1;
    }

    QueryResult* global_results = new QueryResult[base_number];
    
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        sq.encode(test_query + i * vecdim, sq_query_buf, vecdim);

        auto res = sq_flat_search(aligned_sq_base, sq_query_buf, base_number, vecdim, k, global_results);

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
    
    free(aligned_sq_base);
    free(sq_query_buf);
    delete[] global_results;
    delete[] base; 
    delete[] test_query;
    delete[] test_gt;

    return 0;
}