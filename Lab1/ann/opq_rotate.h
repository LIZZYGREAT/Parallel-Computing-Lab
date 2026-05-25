#pragma once
#include "opq_matrix.h"
#include <omp.h>

// 引入底层硬件指令集
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// x86 AVX2 水平求和辅助函数
#if defined(__AVX2__)
inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow  = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, 1);
    vlow  = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}
#endif

inline void opq_rotate(const float* src, float* dst, int d = 96) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // 针对 Kunpeng (AArch64 Neon)
    for (int j = 0; j < d; ++j) {
        // 使用 4 路累加器，打破前后数据依赖
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        // 每次处理 16 个 float
        for (int i = 0; i < d; i += 16) {
            float32x4_t s0 = vld1q_f32(src + i);
            float32x4_t r0 = vld1q_f32(OPQ_R[j] + i);
            sum0 = vfmaq_f32(sum0, r0, s0);

            float32x4_t s1 = vld1q_f32(src + i + 4);
            float32x4_t r1 = vld1q_f32(OPQ_R[j] + i + 4);
            sum1 = vfmaq_f32(sum1, r1, s1);

            float32x4_t s2 = vld1q_f32(src + i + 8);
            float32x4_t r2 = vld1q_f32(OPQ_R[j] + i + 8);
            sum2 = vfmaq_f32(sum2, r2, s2);

            float32x4_t s3 = vld1q_f32(src + i + 12);
            float32x4_t r3 = vld1q_f32(OPQ_R[j] + i + 12);
            sum3 = vfmaq_f32(sum3, r3, s3);
        }
        
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        
        // 硬件级标量归约求和 (AArch64 指令)
        dst[j] = vaddvq_f32(sum0);
    }

#elif defined(__AVX2__)
    // 针对本地 x86 机 (AVX2)
    for (int j = 0; j < d; ++j) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        // 每次处理 16 个 float
        for (int i = 0; i < d; i += 16) {
            // src 可能在调用前未强制对齐，因此使用非对齐加载 _mm256_loadu_ps
            __m256 s0 = _mm256_loadu_ps(src + i);
            // OPQ_R 已使用 alignas(32) 对齐，安全使用快速对齐加载 _mm256_load_ps
            __m256 r0 = _mm256_load_ps(OPQ_R[j] + i); 
            sum0 = _mm256_fmadd_ps(s0, r0, sum0);

            __m256 s1 = _mm256_loadu_ps(src + i + 8);
            __m256 r1 = _mm256_load_ps(OPQ_R[j] + i + 8);
            sum1 = _mm256_fmadd_ps(s1, r1, sum1);
        }
        sum0 = _mm256_add_ps(sum0, sum1);
        dst[j] = hsum_avx2(sum0);
    }

#else
    // 退化到标量实现
    for (int j = 0; j < d; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += src[i] * OPQ_R[j][i];
        }
        dst[j] = sum;
    }
#endif
}

inline void opq_rotate_batch(const float* src, float* dst, size_t n, int d = 96) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        opq_rotate(src + i * d, dst + i * d, d);
    }
}