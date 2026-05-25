#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>
#if !defined(_WIN32)
#include <malloc.h>
#endif

#ifndef ALIGNED_ALLOC_ALIGN
#if defined(__aarch64__) || defined(__ARM_NEON)
constexpr size_t kDefaultAlign = 64;
#elif defined(__AVX512F__)
constexpr size_t kDefaultAlign = 64;
#elif defined(__AVX2__)
constexpr size_t kDefaultAlign = 32;
#else
constexpr size_t kDefaultAlign = 16;
#endif
#else
constexpr size_t kDefaultAlign = ALIGNED_ALLOC_ALIGN;
#endif

inline constexpr size_t simd_min_align() {
#if defined(__aarch64__) || defined(__ARM_NEON)
    return 16;
#elif defined(__AVX2__)
    return 32;
#else
    return 16;
#endif
}

template<typename T>
class AlignedBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
    static constexpr size_t kAlign = kDefaultAlign;

    void release() {
        if (ptr_) {
#if defined(_WIN32)
            _aligned_free(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
        }
        size_ = 0;
    }

public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) { resize(n); }
    ~AlignedBuffer() { release(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        if (n == size_) return;
        release();
        if (n == 0) return;
        void* p = nullptr;
#if defined(_WIN32)
        p = _aligned_malloc(n * sizeof(T), kAlign);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, kAlign, n * sizeof(T)) != 0) p = nullptr;
        if (!p) throw std::bad_alloc();
#endif
        ptr_ = static_cast<T*>(p);
        size_ = n;
    }

    void assign(const std::vector<T>& v) {
        if (v.size() != size_) resize(v.size());
        if (!v.empty()) std::memcpy(ptr_, v.data(), v.size() * sizeof(T));
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    static constexpr size_t alignment() noexcept { return kAlign; }
};
