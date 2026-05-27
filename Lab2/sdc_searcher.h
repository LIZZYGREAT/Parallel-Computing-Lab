#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "profiler.h"
#include "pq_distance.h"
#include "fast_scan_kernel.h"
#include <omp.h>
#include <algorithm>
#include <limits>
#include <vector>

class SDCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;
    const float* base_data;
    int rerank_ratio;
    AlignedBuffer<float> center_dist_table;
    AlignedBuffer<float> coarse_dists;

    template<typename Cmp>
    __attribute__((always_inline)) void score_block_slots(
        const FSBlock& block, const uint16_t* sum_arr,
        float base_dist, float scale, float& threshold,
        std::vector<Candidate>& topk, int rerank_k, Cmp cmp) const {
        #pragma GCC unroll 16
        for (int k = 0; k < 16; ++k) {
            uint32_t id = block.ids[k];
            if (id == 0xFFFFFFFF) continue;
            float approx_dist = base_dist + scale * sum_arr[k];
            if (approx_dist < threshold) {
                topk.push_back({approx_dist, id});
                if (topk.size() >= static_cast<size_t>(rerank_k * 2)) {
                    std::nth_element(topk.begin(), topk.begin() + rerank_k, topk.end(), cmp);
                    topk.resize(rerank_k);
                    threshold = std::max_element(topk.begin(), topk.end(), cmp)->dist;
                }
            }
        }
    }

public:
    SDCSearcher(const IVFPQIndex* idx, const float* base, int ratio = 20)
        : index(idx), base_data(base), rerank_ratio(ratio) {
        center_dist_table.resize(FS_M * FS_K * FS_K);
        coarse_dists.resize(index->n_lists);

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
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
            for (int c = 0; c < index->n_lists; ++c) {
                coarse_cands[c] = {coarse_dists[c], static_cast<uint32_t>(c)};
            }
        }
        {
            MicroProfiler::Timer _t("2_Coarse_Sort");
            std::partial_sort(coarse_cands.begin(), coarse_cands.begin() + nprobe, coarse_cands.end(), cmp_asc);
        }

        int n_threads = omp_get_max_threads();
        std::vector<std::vector<Candidate>> per_thread(n_threads);

        #pragma omp parallel
        {
            SearchWorkspace ws;
            std::vector<Candidate> local_topk;
            local_topk.reserve(rerank_k * 2);
            float local_threshold = std::numeric_limits<float>::max();
            int tid = omp_get_thread_num();

            #pragma omp for schedule(static)
            for (int i = 0; i < nprobe; ++i) {
                int list_id = coarse_cands[i].id;
                const float coarse_dist = coarse_cands[i].dist;
                const InvertedList& cur_list = index->lists[list_id];
                if (cur_list.total_elements == 0) continue;

                const float* cent = &index->ivf_centroids[list_id * index->d];
                {
                    MicroProfiler::Timer _t("3_Compute_Residual");
                    residual_sub_d96(query, cent, ws.residual.data());
                }

                float min_val, max_val;
                {
                    MicroProfiler::Timer _t("4_Quantize_And_LUT");
                    sdc_quantize_and_build_lut(ws.residual.data(), FS_M, index->d_sub,
                                               index->pq_centroids.data(), center_dist_table.data(),
                                               ws.lut_f.data(), ws.query_code.data(),
                                               min_val, max_val);
                }

                float scale = (max_val - min_val) / 255.0f;
                float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
                float base_dist = coarse_dist + FS_M * min_val;
                lut_float_to_u8(ws.lut_f.data(), FS_M * 16, min_val, inv_scale, ws.lut_u8.data());

                {
                    MicroProfiler::Timer _t("5_FastScan_SDC");
                    const FSBlock* blocks = cur_list.blocks.data();
                    const size_t nb = cur_list.blocks.size();
                    auto score_fn = [&](const FSBlock& block, const uint16_t* sums) {
                        score_block_slots(block, sums, base_dist, scale,
                                          local_threshold, local_topk, rerank_k, cmp_asc);
                    };
                    fast_scan_list_batch(blocks, nb, ws.lut_u8.data(), FS_M,
                                         ws.sum_arr.data(), score_fn);
                }
            }

            {
                MicroProfiler::Timer _t("6_Local_TopK_Trim");
                if (local_topk.size() > static_cast<size_t>(rerank_k)) {
                    std::nth_element(local_topk.begin(), local_topk.begin() + rerank_k, local_topk.end(), cmp_asc);
                    local_topk.resize(rerank_k);
                }
            }
            per_thread[tid] = std::move(local_topk);
        }

        std::vector<Candidate> global_topk;
        {
            MicroProfiler::Timer _t("7_Thread_Merge");
            for (auto& v : per_thread) {
                global_topk.insert(global_topk.end(), v.begin(), v.end());
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
            const int n = static_cast<int>(global_topk.size());
            std::vector<uint32_t> ids(n);
            std::vector<float> dists(n);
            for (int i = 0; i < n; ++i) ids[i] = global_topk[i].id;
            rerank_batch_d96(query, base_data, index->d, ids.data(), dists.data(), n);
            for (int i = 0; i < n; ++i) global_topk[i].dist = dists[i];
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
