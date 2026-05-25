#pragma once
#include "opq_matrix.h"
#include <omp.h>

inline void opq_rotate(const float* src, float* dst, int d = 96) {
    for (int j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += src[i] * OPQ_R[j][i];
        }
        dst[j] = sum;
    }
}

inline void opq_rotate_batch(const float* src, float* dst, size_t n, int d = 96) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        opq_rotate(src + i * d, dst + i * d, d);
    }
}
