#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "kmeans.h"
#include "profiler.h"
#include <omp.h>
#include <algorithm>
#include <limits>

class ADCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;

public:
    ADCSearcher(const IVFPQIndex* idx) : index(idx) {}

    std::priority_queue<Candidate> search(const float* query, int top_k, int nprobe) override {
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
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end());
        }
        
        std::vector<Candidate> global_topk;
        
        #pragma omp parallel
        {
            std::vector<Candidate> local_topk;
            local_topk.reserve(top_k * 2);
            float local_threshold = std::numeric_limits<float>::max();

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < nprobe; ++i) {
                int list_id = coarse_cands[i].id;
                const float coarse_dist = coarse_cands[i].dist;
                const InvertedList& cur_list = index->lists[list_id];
                if (cur_list.ids.empty()) continue;

                std::vector<float> lut(index->M * 256);
                std::vector<float> residual_q(index->d);
                {
                    MicroProfiler::Timer _t("3_Compute_Residual");
                    for (int j = 0; j < index->d; ++j) {
                        residual_q[j] = query[j] - index->ivf_centroids[list_id * index->d + j];
                    }
                }
                {
                    MicroProfiler::Timer _t("4_Build_LUT");
                    for (int m = 0; m < index->M; ++m) {
                        const float* sub_query = &residual_q[m * index->d_sub];
                        const float* sub_centers = &index->pq_centroids[m * 256 * index->d_sub];
                        for (int k = 0; k < 256; ++k) {
                            lut[m * 256 + k] = compute_L2_sqr(sub_query, sub_centers + k * index->d_sub, index->d_sub);
                        }
                    }
                }

                size_t list_size = cur_list.ids.size();
                const uint8_t* codes = cur_list.codes.data();
                const uint32_t* ids = cur_list.ids.data();

                {
                    MicroProfiler::Timer _t("5_ADC_Scan");
                    for (size_t idx = 0; idx < list_size; ++idx) {
                        float approx_dist = coarse_dist;
                        const uint8_t* cur_code = codes + idx * index->M;
                        
                        #pragma GCC unroll 16
                        for (int m = 0; m < index->M; ++m) {
                            approx_dist += lut[m * 256 + cur_code[m]];
                        }

                        if (approx_dist < local_threshold) {
                            local_topk.push_back({approx_dist, ids[idx]});
                            if (local_topk.size() >= static_cast<size_t>(top_k * 2)) {
                                std::nth_element(local_topk.begin(), local_topk.begin() + top_k, local_topk.end());
                                local_topk.resize(top_k);
                                auto max_it = std::max_element(local_topk.begin(), local_topk.end());
                                local_threshold = max_it->dist;
                            }
                        }
                    }
                }
            }

            {
                MicroProfiler::Timer _t("6_Local_TopK_Trim");
                if (local_topk.size() > static_cast<size_t>(top_k)) {
                    std::nth_element(local_topk.begin(), local_topk.begin() + top_k, local_topk.end());
                    local_topk.resize(top_k);
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
            if (global_topk.size() > static_cast<size_t>(top_k)) {
                std::nth_element(global_topk.begin(), global_topk.begin() + top_k, global_topk.end());
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
