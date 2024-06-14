#include <chrono>
#include <random>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <immintrin.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdlib>
#include "cmdline.hpp"
#include "distance.hpp"
template<typename T>
class L2 {
public:
    L2(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
        buf_ = new char[K_];
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            volatile auto res = melee::L2Distance(reinterpret_cast<T *>(arr_ + idx), reinterpret_cast<T *>(buf_), K_ / sizeof(T));
            static_cast<void>(res);
        }
    }

private:
    char *arr_;
    char *buf_;
    size_t *indices_;
    size_t K_;
};

template<typename T>
class PrefetchL2 {
public:
    PrefetchL2(char *arr, size_t *indices, size_t K)
            : arr_(arr), indices_(indices), K_(K) {
        buf_ = new char[K_];
    }

    const void operator()(const tbb::blocked_range<size_t> &range) const {
        for (size_t i = range.begin(); i != range.end(); i++) {
            size_t idx = indices_[i];
            size_t next_idx = indices_[i + 1];
            _mm_prefetch(arr_ + next_idx, _MM_HINT_T1);
            volatile auto res = melee::L2Distance(reinterpret_cast<T *>(arr_ + idx), reinterpret_cast<T *>(buf_), K_ / sizeof(T));
            static_cast<void>(res);
        }
    }

private:
    char *arr_;
    char *buf_;
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
    auto foo = get_bandwidth<L2<uint8_t >>(arr, indices, config);
    auto L2_u8_bw = get_bandwidth<L2<uint8_t >>(arr, indices, config);
    auto L2_f16_bw = get_bandwidth<L2<float16>>(arr, indices, config);
    auto L2_f32_bw = get_bandwidth<L2<float>>(arr, indices, config);
    auto PL2_u8_bw = get_bandwidth<PrefetchL2<uint8_t >>(arr, indices, config);
    auto PL2_f16_bw = get_bandwidth<PrefetchL2<float16>>(arr, indices, config);
    auto PL2_f32_bw = get_bandwidth<PrefetchL2<float>>(arr, indices, config);
    spdlog::info("Iterations={} ArrSize={}GB Threads={} BlockSize={} L2_U8={:.1f}GB/s PL2_U8={:.1f}GB/s L2_F16={:.1f}GB/s PL2_F16={:.1f}GB/s L2_F32={:.1f}GB/s PL2_F32={:.1f}GB/s", config.N, config.ARR_SIZE / 1024 / 1024 / 1024, config.T, config.K,
                 L2_u8_bw, PL2_u8_bw, L2_f16_bw, PL2_f16_bw, L2_f32_bw, PL2_f32_bw);

    delete[] arr;
    return 0;
}