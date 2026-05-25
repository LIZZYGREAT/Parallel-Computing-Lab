#pragma once
#include <cstdint>
#include "ivfpq_index.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block,
    const uint8_t* lut_u8,
    int M,
    uint16_t sum_out[16]) {
    uint16x8_t p0_lo = vdupq_n_u16(0), p0_hi = vdupq_n_u16(0);
    uint16x8_t p1_lo = vdupq_n_u16(0), p1_hi = vdupq_n_u16(0);
    uint16x8_t p2_lo = vdupq_n_u16(0), p2_hi = vdupq_n_u16(0);
    uint16x8_t p3_lo = vdupq_n_u16(0), p3_hi = vdupq_n_u16(0);

    int m = 0;
    for (; m + 4 <= M; m += 4) {
        uint8x16_t c0 = vld1q_u8(block.codes[m]);
        uint8x16_t l0 = vld1q_u8(lut_u8 + m * 16);
        uint8x16_t v0 = vqtbl1q_u8(l0, c0);
        p0_lo = vaddw_u8(p0_lo, vget_low_u8(v0));
        p0_hi = vaddw_u8(p0_hi, vget_high_u8(v0));

        uint8x16_t c1 = vld1q_u8(block.codes[m + 1]);
        uint8x16_t l1 = vld1q_u8(lut_u8 + (m + 1) * 16);
        uint8x16_t v1 = vqtbl1q_u8(l1, c1);
        p1_lo = vaddw_u8(p1_lo, vget_low_u8(v1));
        p1_hi = vaddw_u8(p1_hi, vget_high_u8(v1));

        uint8x16_t c2 = vld1q_u8(block.codes[m + 2]);
        uint8x16_t l2 = vld1q_u8(lut_u8 + (m + 2) * 16);
        uint8x16_t v2 = vqtbl1q_u8(l2, c2);
        p2_lo = vaddw_u8(p2_lo, vget_low_u8(v2));
        p2_hi = vaddw_u8(p2_hi, vget_high_u8(v2));

        uint8x16_t c3 = vld1q_u8(block.codes[m + 3]);
        uint8x16_t l3 = vld1q_u8(lut_u8 + (m + 3) * 16);
        uint8x16_t v3 = vqtbl1q_u8(l3, c3);
        p3_lo = vaddw_u8(p3_lo, vget_low_u8(v3));
        p3_hi = vaddw_u8(p3_hi, vget_high_u8(v3));
    }
    for (; m < M; ++m) {
        uint8x16_t c = vld1q_u8(block.codes[m]);
        uint8x16_t l = vld1q_u8(lut_u8 + m * 16);
        uint8x16_t v = vqtbl1q_u8(l, c);
        p0_lo = vaddw_u8(p0_lo, vget_low_u8(v));
        p0_hi = vaddw_u8(p0_hi, vget_high_u8(v));
    }

    uint16x8_t sum_lo = vaddq_u16(vaddq_u16(p0_lo, p1_lo), vaddq_u16(p2_lo, p3_lo));
    uint16x8_t sum_hi = vaddq_u16(vaddq_u16(p0_hi, p1_hi), vaddq_u16(p2_hi, p3_hi));
    vst1q_u16(sum_out, sum_lo);
    vst1q_u16(sum_out + 8, sum_hi);
}

#elif defined(__AVX2__)
#include <immintrin.h>

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block,
    const uint8_t* lut_u8,
    int M,
    uint16_t sum_out[16]) {
    __m256i p0 = _mm256_setzero_si256();
    __m256i p1 = _mm256_setzero_si256();
    __m256i p2 = _mm256_setzero_si256();
    __m256i p3 = _mm256_setzero_si256();

    int m = 0;
    for (; m + 4 <= M; m += 4) {
        __m128i c0 = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[m]));
        __m128i l0 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + m * 16));
        p0 = _mm256_add_epi16(p0, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l0, c0)));

        __m128i c1 = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[m + 1]));
        __m128i l1 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + (m + 1) * 16));
        p1 = _mm256_add_epi16(p1, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l1, c1)));

        __m128i c2 = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[m + 2]));
        __m128i l2 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + (m + 2) * 16));
        p2 = _mm256_add_epi16(p2, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l2, c2)));

        __m128i c3 = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[m + 3]));
        __m128i l3 = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + (m + 3) * 16));
        p3 = _mm256_add_epi16(p3, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l3, c3)));
    }
    for (; m < M; ++m) {
        __m128i c = _mm_load_si128(reinterpret_cast<const __m128i*>(block.codes[m]));
        __m128i l = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_u8 + m * 16));
        p0 = _mm256_add_epi16(p0, _mm256_cvtepu8_epi16(_mm_shuffle_epi8(l, c)));
    }

    __m256i sum = _mm256_add_epi16(
        _mm256_add_epi16(p0, p1),
        _mm256_add_epi16(p2, p3));
    _mm256_store_si256(reinterpret_cast<__m256i*>(sum_out), sum);
}

#else

__attribute__((always_inline)) inline void fast_scan_block_accumulate(
    const FSBlock& block,
    const uint8_t* lut_u8,
    int M,
    uint16_t sum_out[16]) {
    for (int k = 0; k < 16; ++k) sum_out[k] = 0;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < 16; ++k) {
            sum_out[k] += lut_u8[m * 16 + block.codes[m][k]];
        }
    }
}

#endif
