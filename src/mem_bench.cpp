#include <chrono>
#include <random>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <immintrin.h>
#include <stdio.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include "cmdline.hpp"

class Memcpy {
public:
    Memcpy(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
        buf_ = new char[K_];
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            memcpy(buf_, arr_ + idx, K_);
        }
    }

private:
    char *arr_;
    char *buf_;
    size_t *indices_;
    size_t K_;
};

class PrefetchMemcpy {
public:
    PrefetchMemcpy(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
        buf_ = new char[K_];
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            size_t next_idx = indices_[i + 1];
            _mm_prefetch(arr_ + next_idx, _MM_HINT_T1);
            memcpy(buf_, arr_ + idx, K_);
        }
    }

private:
    char *arr_;
    char *buf_;
    size_t *indices_;
    size_t K_;
};

template<typename T>
class Accumulate {
public:
    Accumulate(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            volatile const T res = std::accumulate(reinterpret_cast<T*>(arr_ + idx), reinterpret_cast<T*>(arr_ + idx + K_), static_cast<T>(0));
        }
    }

private:
    char *arr_;
    size_t *indices_;
    size_t K_;
};

template<typename T>
class PrefetchAccumulate {
public:
    PrefetchAccumulate(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            size_t next_idx = indices_[i + 1];
            _mm_prefetch(arr_ + next_idx, _MM_HINT_T1);
            volatile const T res = std::accumulate(reinterpret_cast<T *>(arr_ + idx), reinterpret_cast<T *>(arr_ + idx + K_), static_cast<T>(0));
        }
    }

private:
    char *arr_;
    size_t *indices_;
    size_t K_;
};

template<class F>
double get_bandwidth(char* arr, size_t* indices, melee::MemConfig config){
    auto start = std::chrono::high_resolution_clock::now();
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, config.N),
                              F(arr, indices, config.K), oneapi::tbb::static_partitioner());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double bandwidth = (config.N * config.K) / diff.count() / (1024 * 1024 * 1024);
    return bandwidth;
}

int main(int argc, char *argv[]) {
    // Allocate array

    melee::MemConfig config = melee::MemConfig(argc, argv);
    size_t alignment = 4096;
    char *arr = static_cast<char*>(std::aligned_alloc(alignment, config.ARR_SIZE));
    size_t *indices = new size_t[config.N + 1];
    // Seed random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = 0; i <= config.N; i++) {
        std::uniform_int_distribution<size_t> dis(0, config.ARR_SIZE - config.K);
        size_t idx = dis(gen);
        idx -= idx % config.K;  // Align to K-byte boundary
        indices[i] = idx;
    }

    oneapi::tbb::global_control global_limit(
            oneapi::tbb::global_control::max_allowed_parallelism, config.T);

    // Measure Naive Timing
    auto memcpy_bw = get_bandwidth<Memcpy>(arr, indices, config);
    auto prefetch_memcpy_bw = get_bandwidth<PrefetchMemcpy>(arr, indices, config);
    auto f32_accumulate_bw = get_bandwidth<Accumulate<float>>(arr, indices, config);
    auto f32_prefetch_accumulate_bw = get_bandwidth<PrefetchAccumulate<float>>(arr, indices, config);
    auto u8_accumulate_bw = get_bandwidth<Accumulate<unsigned char>>(arr, indices, config);
    auto u8_prefetch_accumulate_bw = get_bandwidth<PrefetchAccumulate<unsigned char>>(arr, indices, config);
    spdlog::info("Iterations={} Threads={} BlockSize={} Memcpy={:.1f}GB/s PrefetchMemcpy={:.1f}GB/s F32Acc={:.1f}GB/s PrefetchF32Acc={:.1f}GB/s U8Acc={:.1f}GB/s PrefetchU8Acc={:.1f}GB/s", config.N, config.T,
                 config.K, memcpy_bw, prefetch_memcpy_bw, f32_accumulate_bw, f32_prefetch_accumulate_bw, u8_accumulate_bw, u8_prefetch_accumulate_bw);
    delete[] arr;
    return 0;
}