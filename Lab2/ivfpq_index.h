#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <omp.h>
#include "kmeans.h"
#include "profiler.h"
#include "simd_l2.h"
#include "aligned_alloc.h"

constexpr int FS_D = 96;
constexpr int FS_M = 32;
constexpr int FS_K = 16;
constexpr int FS_D_SUB = 3;

struct alignas(16) FSBlock {
    uint8_t codes[FS_M][16];
    uint32_t ids[16];
};

struct InvertedList {
    std::vector<FSBlock> blocks;
    size_t total_elements = 0;
};

struct TempVec {
    uint8_t code[FS_M];
    uint32_t id;
};

class IVFPQIndex {
public:
    int d = FS_D;
    int n_lists;
    int M = FS_M;
    int K = FS_K;
    int d_sub = FS_D_SUB;

    AlignedVector<float> ivf_centroids; 
    AlignedVector<float> pq_centroids;  
    std::vector<InvertedList> lists;  

    IVFPQIndex(int dim = 96, int nlist = 1024) 
        : d(dim), n_lists(nlist) {
        ivf_centroids.resize(n_lists * d);
        pq_centroids.resize(M * K * d_sub);
        lists.resize(n_lists);
    }

    void build(const float* data, size_t n) {
        std::cerr << "[IVFPQ Build] d=" << d << ", n_lists=" << n_lists 
                  << ", M=" << M << ", K=" << K << ", d_sub=" << d_sub << "\n";

        {
            MicroProfiler::Timer _t("1_Train_IVF");
            KMeans ivf_km(d, n_lists);
            ivf_km.train(data, n);
            std::memcpy(ivf_centroids.data(), ivf_km.centroids.data(), n_lists * d * sizeof(float));
        }

        std::vector<int> assign(n);
        std::vector<float> residuals(n * d);

        {
            MicroProfiler::Timer _t("2_Compute_Residuals");
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < n_lists; ++c) {
                    float dist = compute_L2_sqr(data + i * d, &ivf_centroids[c * d], d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assign[i] = best_c;
                for (int j = 0; j < d; ++j) {
                    residuals[i * d + j] = data[i * d + j] - ivf_centroids[best_c * d + j];
                }
            }
        }

        {
            MicroProfiler::Timer _t("3_Train_PQ");
            #pragma omp parallel for schedule(dynamic)
            for (int m = 0; m < M; ++m) {
                std::vector<float> sub_data(n * d_sub);
                for (size_t i = 0; i < n; ++i) {
                    for (int j = 0; j < d_sub; ++j) {
                        sub_data[i * d_sub + j] = residuals[i * d + m * d_sub + j];
                    }
                }
                KMeans pq_km(d_sub, K);
                pq_km.train(sub_data.data(), n);
                std::memcpy(&pq_centroids[m * K * d_sub], pq_km.centroids.data(), K * d_sub * sizeof(float));
            }
        }

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::vector<TempVec>>> thread_local_lists(num_threads, std::vector<std::vector<TempVec>>(n_lists));

        {
            MicroProfiler::Timer _t("4_Encode_PQ");
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    TempVec tv;
                    tv.id = i;
                    for (int m = 0; m < M; ++m) {
                        const float* sub_res = &residuals[i * d + m * d_sub];
                        const float* sub_cents = &pq_centroids[m * K * d_sub];
                        
                        float min_dist = std::numeric_limits<float>::max();
                        uint8_t best_k = 0;
                        for (int k = 0; k < K; ++k) {
                            float dist = compute_L2_sqr(sub_res, sub_cents + k * d_sub, d_sub);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_k = static_cast<uint8_t>(k);
                            }
                        }
                        tv.code[m] = best_k;
                    }
                    thread_local_lists[tid][assign[i]].push_back(tv);
                }
            }
        }

        {
            MicroProfiler::Timer _t("5_Interleave_Blocks");
            #pragma omp parallel for schedule(dynamic)
            for (int c = 0; c < n_lists; ++c) {
                std::vector<TempVec> merged_list;
                for (int t = 0; t < num_threads; ++t) {
                    merged_list.insert(merged_list.end(), thread_local_lists[t][c].begin(), thread_local_lists[t][c].end());
                }

                size_t num = merged_list.size();
                lists[c].total_elements = num;
                
                size_t num_blocks = (num + 15) / 16;
                lists[c].blocks.resize(num_blocks);

                for (size_t b = 0; b < num_blocks; ++b) {
                    FSBlock& block = lists[c].blocks[b];
                    for (int k = 0; k < 16; ++k) {
                        size_t idx = b * 16 + k;
                        if (idx < num) {
                            block.ids[k] = merged_list[idx].id;
                            for (int m = 0; m < M; ++m) {
                                block.codes[m][k] = merged_list[idx].code[m];
                            }
                        } else {
                            block.ids[k] = 0xFFFFFFFF; 
                            for (int m = 0; m < M; ++m) {
                                block.codes[m][k] = 0;     
                            }
                        }
                    }
                }
            }
        }
        std::cerr << "[IVFPQ Build] Done.\n";
    }
};