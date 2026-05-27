#pragma once
#include "searcher.h"
#include "ivfpq_index.h"
#include "profiler.h"
#include "pq_distance.h"
#include "fast_scan_kernel.h"
#include "thread_pool.h"
#include "search_buffers.h"
#include <algorithm>
#include <limits>
#include <vector>

class ADCSearcher : public BaseSearcher {
private:
    const IVFPQIndex* index;
    const float* base_data;
    int rerank_ratio;
    AlignedBuffer<float> coarse_dists;
    SearchBuffers tls_;

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

    template<typename Cmp>
    void scan_probe_list(int list_id, float coarse_dist, const float* query,
                         SearchWorkspace& ws, float& threshold,
                         std::vector<Candidate>& topk, int rerank_k, Cmp cmp) const {
        const InvertedList& cur_list = index->lists[list_id];
        if (cur_list.total_elements == 0) return;

        const float* cent = &index->ivf_centroids[list_id * index->d];
        residual_sub_d96(query, cent, ws.residual);

        float min_val, max_val;
        pq_build_adc_lut(ws.residual, FS_M, index->d_sub,
                         index->pq_centroids.data(), ws.lut_f,
                         min_val, max_val);
        float scale = (max_val - min_val) / 255.0f;
        float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        float base_dist = coarse_dist + FS_M * min_val;
        lut_float_to_u8(ws.lut_f, FS_M * 16, min_val, inv_scale, ws.lut_u8);

        const FSBlock* blocks = cur_list.blocks.data();
        const size_t nb = cur_list.blocks.size();
        auto score_fn = [&](const FSBlock& block, const uint16_t* sums) {
            score_block_slots(block, sums, base_dist, scale, threshold, topk, rerank_k, cmp);
        };
        fast_scan_list_batch(blocks, nb, ws.lut_u8, FS_M, ws.sum_arr, score_fn);
    }

public:
    ADCSearcher(const IVFPQIndex* idx, const float* base, int ratio = 20)
        : index(idx), base_data(base), rerank_ratio(ratio) {
        coarse_dists.resize(index->n_lists);
    }

    std::priority_queue<Candidate> search(const float* query, int top_k, int nprobe) override {
        int rerank_k = top_k * rerank_ratio;
        auto cmp_asc = [](const Candidate& a, const Candidate& b) {
            return a.dist < b.dist;
        };

        const int nt = tp::get_num_threads();
        tls_.prepare(nt, index->n_lists, rerank_k);

        {
            MicroProfiler::Timer _t("1_Coarse_Dist");
            compute_all_L2_sqr_d96(query, index->ivf_centroids.data(), index->n_lists, coarse_dists.data());
            for (int c = 0; c < index->n_lists; ++c) {
                tls_.coarse_cands[static_cast<size_t>(c)] = {coarse_dists[c], static_cast<uint32_t>(c)};
            }
        }
        {
            MicroProfiler::Timer _t("2_Coarse_Sort");
            std::partial_sort(tls_.coarse_cands.begin(), tls_.coarse_cands.begin() + nprobe,
                              tls_.coarse_cands.end(), cmp_asc);
        }

        {
            MicroProfiler::Timer _t("3_Probe_Scan");
            auto run_probes = [&](int tid, int tcount) {
                SearchWorkspace& ws = tls_.workspaces[static_cast<size_t>(tid)];
                std::vector<Candidate>& local_topk = tls_.per_thread_topk[static_cast<size_t>(tid)];
                local_topk.clear();
                float local_threshold = std::numeric_limits<float>::max();
                const size_t chunk = (static_cast<size_t>(nprobe) + static_cast<size_t>(tcount) - 1)
                                   / static_cast<size_t>(tcount);
                const size_t p0 = static_cast<size_t>(tid) * chunk;
                const size_t p1 = std::min(static_cast<size_t>(nprobe), p0 + chunk);
                for (size_t pi = p0; pi < p1; ++pi) {
                    int list_id = static_cast<int>(tls_.coarse_cands[pi].id);
                    scan_probe_list(list_id, tls_.coarse_cands[pi].dist, query, ws,
                                    local_threshold, local_topk, rerank_k, cmp_asc);
                }
                if (local_topk.size() > static_cast<size_t>(rerank_k)) {
                    std::nth_element(local_topk.begin(), local_topk.begin() + rerank_k,
                                     local_topk.end(), cmp_asc);
                    local_topk.resize(rerank_k);
                }
            };

            if (tp::should_parallel(static_cast<size_t>(nprobe), nt)) {
                tp::parallel_region([&](int tid) { run_probes(tid, nt); });
            } else {
                run_probes(0, 1);
            }
        }

        std::vector<Candidate>& global_topk = tls_.merged;
        global_topk.clear();
        {
            MicroProfiler::Timer _t("7_Thread_Merge");
            for (int t = 0; t < nt; ++t) {
                const auto& v = tls_.per_thread_topk[static_cast<size_t>(t)];
                global_topk.insert(global_topk.end(), v.begin(), v.end());
            }
        }

        {
            MicroProfiler::Timer _t("8_Global_Merge");
            if (global_topk.size() > static_cast<size_t>(rerank_k)) {
                std::nth_element(global_topk.begin(), global_topk.begin() + rerank_k,
                                 global_topk.end(), cmp_asc);
                global_topk.resize(rerank_k);
            }
        }

        {
            MicroProfiler::Timer _t("8.5_Re_Rank");
            const int n = static_cast<int>(global_topk.size());
            for (int i = 0; i < n; ++i) tls_.rerank_ids[static_cast<size_t>(i)] = global_topk[static_cast<size_t>(i)].id;
            rerank_batch_d96(query, base_data, index->d, tls_.rerank_ids.data(), tls_.rerank_dists.data(), n);
            for (int i = 0; i < n; ++i) global_topk[static_cast<size_t>(i)].dist = tls_.rerank_dists[static_cast<size_t>(i)];
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
