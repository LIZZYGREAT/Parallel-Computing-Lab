#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cstring>


constexpr int PQ_D = 96;          // 原始向量维度
constexpr int PQ_M = 24;          // 子空间数量 24
constexpr int PQ_K = 256;         // 每个子空间的聚类中心数 
constexpr int PQ_D_SUB = 4;       // 每个子空间的维度
constexpr int PREFETCH_DIST = 16; // 软件预取距离 
constexpr int TOP_C = 200;        


struct alignas(32) PQCode {
    uint8_t code[PQ_M];
    uint8_t padding[32 - PQ_M]; 
};

class SubspaceKMeans {
public:
    int d; 
    int k; 
    std::vector<float> centroids; 

    SubspaceKMeans(int dim = PQ_D_SUB, int num_clusters = PQ_K) : d(dim), k(num_clusters) {
        centroids.resize(k * d, 0.0f);
    }

    void train(const float* train_data, size_t n, int max_iter = 20) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());

        for (int c = 0; c < k; ++c) {
            size_t idx = indices[c];
            for (int j = 0; j < d; ++j) {
                centroids[c * d + j] = train_data[idx * d + j];
            }
        }

        std::vector<int> assigns(n, 0);
        std::vector<float> new_centroids(k * d, 0.0f);
        std::vector<int> counts(k, 0);

        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < k; ++c) {
                    float dist = 0.0f;
                    for (int j = 0; j < d; ++j) {
                        float diff = train_data[i * d + j] - centroids[c * d + j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assigns[i] = best_c;
                counts[best_c]++;
                for (int j = 0; j < d; ++j) {
                    new_centroids[best_c * d + j] += train_data[i * d + j];
                }
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] == 0) {
                    size_t rand_idx = rand() % n;
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = train_data[rand_idx * d + j];
                    }
                } else {
                    for (int j = 0; j < d; ++j) {
                        centroids[c * d + j] = new_centroids[c * d + j] / counts[c];
                    }
                }
            }
        }
    }
};

class PQQuantizer {
public:
    std::vector<SubspaceKMeans> subspaces;

    PQQuantizer() {
        subspaces.resize(PQ_M, SubspaceKMeans(PQ_D_SUB, PQ_K));
    }

    void train(const float* base_data, size_t n) {
        std::cerr << "[PQ Info] Starting KMeans training for " << PQ_M << " subspaces...\n";
        
        std::vector<std::vector<float>> sub_train_data(PQ_M, std::vector<float>(n * PQ_D_SUB));
        
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < PQ_M; ++m) {
            for (size_t i = 0; i < n; ++i) {
                for (int j = 0; j < PQ_D_SUB; ++j) {
                    sub_train_data[m][i * PQ_D_SUB + j] = base_data[i * PQ_D + m * PQ_D_SUB + j];
                }
            }
            subspaces[m].train(sub_train_data[m].data(), n);
        }
        std::cerr << "[PQ Info] KMeans training completed.\n";
    }

    void encode_batch(const float* base_data, PQCode* output_codes, size_t n) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (int m = 0; m < PQ_M; ++m) {
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                
                const float* sub_vec = base_data + i * PQ_D + m * PQ_D_SUB;
                const float* centroids = subspaces[m].centroids.data();

                for (int c = 0; c < PQ_K; ++c) {
                    float dist = 0.0f;
                    for (int j = 0; j < PQ_D_SUB; ++j) {
                        float diff = sub_vec[j] - centroids[c * PQ_D_SUB + j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                output_codes[i].code[m] = static_cast<uint8_t>(best_c);
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const PQCode* base_codes, 
        const float* original_base, 
        const float* query, 
        size_t base_number, 
        size_t top_k
    ) {
        alignas(32) float lut[PQ_M][PQ_K]; 
        
        for (int m = 0; m < PQ_M; ++m) {
            const float* sub_query = query + m * PQ_D_SUB;
            const float* centroids = subspaces[m].centroids.data();
            
            for (int c = 0; c < PQ_K; ++c) {
                float ip = 0.0f;
                for (int j = 0; j < PQ_D_SUB; ++j) {
                    ip += sub_query[j] * centroids[c * PQ_D_SUB + j];
                }
                lut[m][c] = ip;
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> candidate_pq;

        #pragma omp parallel
        {
            std::priority_queue<std::pair<float, uint32_t>> local_pq;
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < base_number; ++i) {
                if (i + PREFETCH_DIST < base_number) {
                    __builtin_prefetch(base_codes + i + PREFETCH_DIST, 0, 0);
                }

                float total_ip = 0.0f;
                const uint8_t* code = base_codes[i].code;

                #pragma GCC unroll 24
                for (int m = 0; m < PQ_M; ++m) {
                    total_ip += lut[m][code[m]];
                }

                float final_dist = 1.0f - total_ip;

                // 维护容量为 TOP_C 的局部候选池
                if (local_pq.size() < TOP_C) {
                    local_pq.push({final_dist, i});
                } else if (final_dist < local_pq.top().first) {
                    local_pq.pop();
                    local_pq.push({final_dist, i});
                }
            }

            #pragma omp critical
            {
                while (!local_pq.empty()) {
                    auto top = local_pq.top();
                    local_pq.pop();
                    if (candidate_pq.size() < TOP_C) {
                        candidate_pq.push(top);
                    } else if (top.first < candidate_pq.top().first) {
                        candidate_pq.pop();
                        candidate_pq.push(top);
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, uint32_t>> final_pq;
        
        while(!candidate_pq.empty()) {
            uint32_t id = candidate_pq.top().second;
            candidate_pq.pop();
            
            float exact_ip = 0.0f;
            const float* base_vec = original_base + id * PQ_D;

            for(int d = 0; d < PQ_D; ++d) {
                exact_ip += query[d] * base_vec[d];
            }
            float exact_dist = 1.0f - exact_ip;

            if (final_pq.size() < top_k) {
                final_pq.push({exact_dist, id});
            } else if (exact_dist < final_pq.top().first) {
                final_pq.pop();
                final_pq.push({exact_dist, id});
            }
        }

        return final_pq;
    }
};