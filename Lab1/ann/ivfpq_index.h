#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <limits>
#include <cstring>
#include <algorithm>
#include <queue>

#include "kmeans.h"
#include "profiler.h"

struct InvertedList {
    std::vector<uint32_t> ids;       // 原始向量 ID
    std::vector<uint8_t> codes;      // PQ 量化编码，总长度为 ids.size() * M
};

class IVFPQIndex {
public:
    int d;           // 原始向量维度 (例如 96)
    int n_lists;     // IVF 粗聚类中心数
    int M;           // PQ 子空间划分数量
    int d_sub;       // 每个子空间的维度 (d / M)
    int K_pq = 256;  // PQ 局部聚类中心数 (固定为 256，映射至 uint8_t)

    std::vector<float> ivf_centroids;             // 大小: n_lists * d
    std::vector<float> pq_centroids;              // 大小: M * 256 * d_sub
    std::vector<InvertedList> lists;              // 大小: n_lists

    IVFPQIndex(int dim, int num_lists, int m) 
        : d(dim), n_lists(num_lists), M(m), d_sub(dim / m) {
        lists.resize(n_lists);
        ivf_centroids.resize(n_lists * d, 0.0f);
        pq_centroids.resize(M * K_pq * d_sub, 0.0f);
    }

    void build(const float* base_data, size_t n) {
        if (n == 0) return;
        std::cerr << "[IVFPQ] Starting index build for " << n << " vectors.\n";

        // ==========================================
        // 阶段 1: IVF 粗聚类与残差计算
        // ==========================================
        std::cerr << "[IVFPQ] Training IVF centroids...\n";
        {
            MicroProfiler::Timer _t("Build_1_IVF_KMeans");
            KMeans ivf_kmeans(d, n_lists);
            ivf_kmeans.train(base_data, n, 15);
            std::memcpy(ivf_centroids.data(), ivf_kmeans.centroids.data(), n_lists * d * sizeof(float));
        }

        std::vector<int> assign(n, 0);
        std::vector<float> residuals(n * d, 0.0f);

        std::cerr << "[IVFPQ] Computing residuals...\n";
        {
            MicroProfiler::Timer _t("Build_2_IVF_Assign");
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* current_data = &base_data[i * d];
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;

                for (int c = 0; c < n_lists; ++c) {
                    float dist = compute_L2_sqr(current_data, &ivf_centroids[c * d], d);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assign[i] = best_c;

                for (int j = 0; j < d; ++j) {
                    residuals[i * d + j] = current_data[j] - ivf_centroids[best_c * d + j];
                }
            }
        }

        // ==========================================
        // 阶段 2: PQ 子空间训练
        // ==========================================
        std::cerr << "[IVFPQ] Training PQ sub-quantizers...\n";
        std::vector<std::vector<float>> sub_train_data(M, std::vector<float>(n * d_sub));
        
        {
            MicroProfiler::Timer _t("Build_3_PQ_Reorder");
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (int m = 0; m < M; ++m) {
                    for (int j = 0; j < d_sub; ++j) {
                        sub_train_data[m][i * d_sub + j] = residuals[i * d + m * d_sub + j];
                    }
                }
            }
        }

        {
            MicroProfiler::Timer _t("Build_4_PQ_KMeans");
            #pragma omp parallel for schedule(dynamic)
            for (int m = 0; m < M; ++m) {
                KMeans pq_km(d_sub, K_pq);
                pq_km.train(sub_train_data[m].data(), n, 20);
                std::memcpy(&pq_centroids[m * K_pq * d_sub], pq_km.centroids.data(), K_pq * d_sub * sizeof(float));
            }
        }

        // ==========================================
        // 阶段 3: 量化编码与倒排桶并行填充
        // ==========================================
        std::cerr << "[IVFPQ] Quantizing and populating inverted lists...\n";
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<InvertedList>> local_lists(num_threads, std::vector<InvertedList>(n_lists));

        {
            MicroProfiler::Timer _t("Build_5_Encode");
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint8_t> local_code(M);

                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    int list_idx = assign[i];
                    const float* res_vec = &residuals[i * d];

                    for (int m = 0; m < M; ++m) {
                        const float* sub_res = res_vec + m * d_sub;
                        const float* sub_centroids = &pq_centroids[m * K_pq * d_sub];
                        
                        float min_dist = std::numeric_limits<float>::max();
                        uint8_t best_pq = 0;

                        for (int k = 0; k < K_pq; ++k) {
                            float dist = compute_L2_sqr(sub_res, sub_centroids + k * d_sub, d_sub);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_pq = static_cast<uint8_t>(k);
                            }
                        }
                        local_code[m] = best_pq;
                    }

                    local_lists[tid][list_idx].ids.push_back(static_cast<uint32_t>(i));
                    local_lists[tid][list_idx].codes.insert(
                        local_lists[tid][list_idx].codes.end(),
                        local_code.begin(), local_code.end()
                    );
                }
            }
        }

        {
            MicroProfiler::Timer _t("Build_6_List_Merge");
            for (int c = 0; c < n_lists; ++c) {
                size_t total_ids = 0;
                size_t total_codes = 0;
                for (int t = 0; t < num_threads; ++t) {
                    total_ids += local_lists[t][c].ids.size();
                    total_codes += local_lists[t][c].codes.size();
                }

                lists[c].ids.reserve(total_ids);
                lists[c].codes.reserve(total_codes);

                for (int t = 0; t < num_threads; ++t) {
                    lists[c].ids.insert(lists[c].ids.end(), local_lists[t][c].ids.begin(), local_lists[t][c].ids.end());
                    lists[c].codes.insert(lists[c].codes.end(), local_lists[t][c].codes.begin(), local_lists[t][c].codes.end());
                }
            }
        }

        std::cerr << "[IVFPQ] Index build completed.\n";
    }

};