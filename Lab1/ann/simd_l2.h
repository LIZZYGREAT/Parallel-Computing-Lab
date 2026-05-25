#pragma once
#include <cstdint>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

__attribute__((always_inline)) inline float compute_l2_sqr_d3(const float* a, const float* b) {
    float d0 = a[0] - b[0];
    float d1 = a[1] - b[1];
    float d2 = a[2] - b[2];
    return d0 * d0 + d1 * d1 + d2 * d2;
}

__attribute__((always_inline)) inline float compute_l2_sqr_d96(const float* a, const float* b) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (int i = 0; i < 96; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t df0 = vsubq_f32(va0, vb0);
        s0 = vmlaq_f32(s0, df0, df0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t df1 = vsubq_f32(va1, vb1);
        s1 = vmlaq_f32(s1, df1, df1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t df2 = vsubq_f32(va2, vb2);
        s2 = vmlaq_f32(s2, df2, df2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t df3 = vsubq_f32(va3, vb3);
        s3 = vmlaq_f32(s3, df3, df3);
    }
    s0 = vaddq_f32(s0, s1);
    s2 = vaddq_f32(s2, s3);
    s0 = vaddq_f32(s0, s2);
    float32x2_t s2l = vadd_f32(vget_low_f32(s0), vget_high_f32(s0));
    return vget_lane_f32(vpadd_f32(s2l, s2l), 0);

#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (int i = 0; i < 96; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 df0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(df0, df0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 df1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(df1, df1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 df2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(df2, df2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 df3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(df3, df3, acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);

#else
    float dist = 0.0f;
    for (int i = 0; i < 96; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
#endif
}

__attribute__((always_inline)) inline float compute_l2_sqr_generic(const float* a, const float* b, int d) {
    float dist = 0.0f;
    int i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (; i <= d - 16; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t df0 = vsubq_f32(va0, vb0);
        s0 = vmlaq_f32(s0, df0, df0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t df1 = vsubq_f32(va1, vb1);
        s1 = vmlaq_f32(s1, df1, df1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t df2 = vsubq_f32(va2, vb2);
        s2 = vmlaq_f32(s2, df2, df2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t df3 = vsubq_f32(va3, vb3);
        s3 = vmlaq_f32(s3, df3, df3);
    }
    for (; i <= d - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t df = vsubq_f32(va, vb);
        s0 = vmlaq_f32(s0, df, df);
    }
    s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float32x2_t s2l = vadd_f32(vget_low_f32(s0), vget_high_f32(s0));
    dist = vget_lane_f32(vpadd_f32(s2l, s2l), 0);

#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i <= d - 32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 df0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(df0, df0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 df1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(df1, df1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 df2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(df2, df2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        __m256 df3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(df3, df3, acc3);
    }
    for (; i <= d - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 df = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(df, df, acc0);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    dist = _mm_cvtss_f32(s);
#endif

    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

__attribute__((always_inline)) inline float compute_L2_sqr(const float* a, const float* b, int d) {
    if (d == 96) return compute_l2_sqr_d96(a, b);
    if (d == 3) return compute_l2_sqr_d3(a, b);
    return compute_l2_sqr_generic(a, b, d);
}
