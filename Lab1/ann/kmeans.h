#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <random>
#include <algorithm>
#include "simd_l2.h"

class KMeans {
public:
    int d; // 向量维度
    int k; // 聚类中心数量
    std::vector<float> centroids; // 聚类中心，大小为 k * d

    KMeans(int dim, int num_clusters) : d(dim), k(num_clusters) {
        centroids.resize(k * d, 0.0f);
    }

    void train(const float* train_data, size_t n, int max_iter = 20) {
        if (n == 0 || k == 0) return;

        // 1. 初始化质心：随机选择 k 个不重复的数据点
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        
        std::mt19937 rng(42); 
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int c = 0; c < k; ++c) {
            size_t idx = indices[c];
            std::memcpy(&centroids[c * d], &train_data[idx * d], d * sizeof(float));
        }

        int max_threads = omp_get_max_threads();
        
        // 申请线程局部内存，展平为一维数组，通过 tid * k * d 索引
        std::vector<float> local_new_centroids(max_threads * k * d, 0.0f);
        std::vector<int> local_counts(max_threads * k, 0);

        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(local_new_centroids.begin(), local_new_centroids.end(), 0.0f);
            std::fill(local_counts.begin(), local_counts.end(), 0);

            // E-步与局部的 M-步：并行计算距离并将数据点累加到线程局部的累加器中
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float* my_centroids_sum = &local_new_centroids[tid * k * d];
                int* my_counts = &local_counts[tid * k];

                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    const float* current_data = &train_data[i * d];
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;

                    for (int c = 0; c < k; ++c) {
                        float dist = compute_L2_sqr(current_data, &centroids[c * d], d);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_c = c;
                        }
                    }

                    // 写入线程局部累加器
                    my_counts[best_c]++;
                    for (int j = 0; j < d; ++j) {
                        my_centroids_sum[best_c * d + j] += current_data[j];
                    }
                }
            } 

            // 全局 M-步：汇总所有线程的局部累加器，计算均值
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < k; ++c) {
                int total_count = 0;
                std::vector<float> global_sum(d, 0.0f);

                for (int t = 0; t < max_threads; ++t) {
                    total_count += local_counts[t * k + c];
                    for (int j = 0; j < d; ++j) {
                        global_sum[j] += local_new_centroids[t * k * d + c * d + j];
                    }
                }

                if (total_count == 0) {
                    size_t rand_idx = rand() % n;
                    std::memcpy(&centroids[c * d], &train_data[rand_idx * d], d * sizeof(float));
                } else {
                    // 更新质心坐标
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = global_sum[j] / total_count;
                    }
                }
            }
        }
    }
};