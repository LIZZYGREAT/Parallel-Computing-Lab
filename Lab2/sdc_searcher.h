#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "kmeans.h"
#include "profiler.h"
#include "simd_l2.h"
#include "fast_scan_kernel.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

class SDCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;
    const float* base_data;
    int rerank_ratio;
    std::vector<float> center_dist_table;

public:
    SDCSearcher(const IVFPQIndex* idx, const float* base, int ratio = 10) 
        : index(idx), base_data(base), rerank_ratio(ratio) {
        center_dist_table.resize(FS_M * FS_K * FS_K, 0.0f);

        MicroProfiler::Timer _t("Init_SDC_Table");
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < FS_M; ++m) {
            for (int i = 0; i < FS_K; ++i) {
                for (int j = 0; j < FS_K; ++j) {
                    const float* c_i = &index->pq_centroids[m * FS_K * FS_D_SUB + i * FS_D_SUB];
                    const float* c_j = &index->pq_centroids[m * FS_K * FS_D_SUB + j * FS_D_SUB];
                    center_dist_table[m * FS_K * FS_K + i * FS_K + j] = compute_L2_sqr(c_i, c_j, FS_D_SUB);
                }
            }
        }
    }

    std::priority_queue<Candidate> search(const float* query, int top_k, int nprobe) override {
        int rerank_k = top_k * rerank_ratio;

        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        };

        std::vector<Candidate> coarse_cands(index->n_lists);
        {
            MicroProfiler::Timer _t("1_Coarse_Dist");
            for (int c = 0; c < index->n_lists; ++c) {
                float dist = compute_L2_sqr(query, &index->ivf_centroids[c * index->d], index->d);
                coarse_cands[c] = {dist, static_cast<uint32_t>(c)};
            }
        }
        {
            MicroProfiler::Timer _t("2_Coarse_Sort");
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(), cmp_asc);
        }
        
        std::vector<Candidate> global_topk;
        
        #pragma omp parallel
        {
            std::vector<Candidate> local_topk;
            local_topk.reserve(rerank_k * 2);
            float local_threshold = std::numeric_limits<float>::max();

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < nprobe; ++i) {
                int list_id = coarse_cands[i].id;
                const float coarse_dist = coarse_cands[i].dist;
                const InvertedList& cur_list = index->lists[list_id];
                if (cur_list.total_elements == 0) continue;

                std::vector<uint8_t> query_code(FS_M);
                std::vector<float> residual_q(index->d);
                {
                    MicroProfiler::Timer _t("3_Compute_Residual");
                    for (int j = 0; j < index->d; ++j) {
                        residual_q[j] = query[j] - index->ivf_centroids[list_id * index->d + j];
                    }
                }
                {
                    MicroProfiler::Timer _t("4_Quantize_Query");
                    for (int m = 0; m < FS_M; ++m) {
                        const float* sub_query = &residual_q[m * index->d_sub];
                        const float* sub_centers = &index->pq_centroids[m * FS_K * index->d_sub];
                        
                        float min_dist = std::numeric_limits<float>::max();
                        uint8_t best_pq = 0;
                        for (int k = 0; k < FS_K; ++k) {
                            float dist = compute_L2_sqr(sub_query, sub_centers + k * index->d_sub, index->d_sub);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_pq = static_cast<uint8_t>(k);
                            }
                        }
                        query_code[m] = best_pq;
                    }
                }

                alignas(64) float lut_f[FS_M * 16];
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                
                {
                    MicroProfiler::Timer _t("4.5_Build_LUT");
                    for (int m = 0; m < FS_M; ++m) {
                        uint8_t q_code = query_code[m];
                        for (int k = 0; k < FS_K; ++k) {
                            float dist = center_dist_table[m * FS_K * FS_K + q_code * FS_K + k];
                            lut_f[m * 16 + k] = dist;
                            min_val = std::min(min_val, dist);
                            max_val = std::max(max_val, dist);
                        }
                    }
                }

                float scale = (max_val - min_val) / 255.0f;
                float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
                float base_dist = coarse_dist + FS_M * min_val;

                alignas(64) uint8_t lut_u8[FS_M * 16];
                for (int j = 0; j < FS_M * 16; ++j) {
                    lut_u8[j] = static_cast<uint8_t>((lut_f[j] - min_val) * inv_scale);
                }

                {
                    MicroProfiler::Timer _t("5_FastScan_SDC");
                    for (size_t b = 0; b < cur_list.blocks.size(); ++b) {
                        const FSBlock& block = cur_list.blocks[b];
                        alignas(32) uint16_t sum_arr[16];
                        fast_scan_block_accumulate(block, lut_u8, FS_M, sum_arr);

                        for (int k = 0; k < 16; ++k) {
                            uint32_t id = block.ids[k];
                            if (id == 0xFFFFFFFF) continue; 

                            float approx_dist = base_dist + scale * sum_arr[k];

                            if (approx_dist < local_threshold) {
                                local_topk.push_back({approx_dist, id});
                                if (local_topk.size() >= static_cast<size_t>(rerank_k * 2)) {
                                    std::nth_element(local_topk.begin(), local_topk.begin() + rerank_k, local_topk.end(), cmp_asc);
                                    local_topk.resize(rerank_k);
                                    auto max_it = std::max_element(local_topk.begin(), local_topk.end(), cmp_asc);
                                    local_threshold = max_it->dist;
                                }
                            }
                        }
                    }
                }
            }

            {
                MicroProfiler::Timer _t("6_Local_TopK_Trim");
                if (local_topk.size() > static_cast<size_t>(rerank_k)) {
                    std::nth_element(local_topk.begin(), local_topk.begin() + rerank_k, local_topk.end(), cmp_asc);
                    local_topk.resize(rerank_k);
                }
            }

            {
                MicroProfiler::Timer _t("7_Thread_Merge");
                #pragma omp critical
                {
                    global_topk.insert(global_topk.end(), local_topk.begin(), local_topk.end());
                }
            }
        }

        {
            MicroProfiler::Timer _t("8_Global_Merge");
            if (global_topk.size() > static_cast<size_t>(rerank_k)) {
                std::nth_element(global_topk.begin(), global_topk.begin() + rerank_k, global_topk.end(), cmp_asc);
                global_topk.resize(rerank_k);
            }
        }
        
        {
            MicroProfiler::Timer _t("8.5_Re_Rank");
            for (auto& cand : global_topk) {
                uint32_t id = cand.id;
                const float* exact_vec = base_data + id * index->d;
                cand.dist = compute_L2_sqr(query, exact_vec, index->d);
            }
            
            if (global_topk.size() > static_cast<size_t>(top_k)) {
                std::nth_element(global_topk.begin(), global_topk.begin() + top_k, global_topk.end(), cmp_asc);
                global_topk.resize(top_k);
            }
        }

        std::priority_queue<Candidate> final_pq;
        {
            MicroProfiler::Timer _t("9_Build_Result");
            for (const auto& cand : global_topk) {
                final_pq.push(cand);
            }
        }
        return final_pq;
    }
};