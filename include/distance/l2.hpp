//
// Created by juelin on 6/5/24.
//

#ifndef MELEE_L2_HPP
#define MELEE_L2_HPP
#include "util.hpp"
#include <cpuid.h>
#include <immintrin.h>
#include <type_traits>

namespace melee {
    namespace impl {

        template <typename dist_t, typename data_t, int scale,
                int dimension = INT32_MAX>
        __force_inline__ dist_t L2DistanceResidual(const data_t *pVect1,
                                                   const data_t *pVect2, int dim) {
            dist_t dist{0};

#pragma unroll
            for (int i = get_main<scale, dimension>(dim);
                 i < get_all<scale, dimension>(dim); i += 1) {
                const dist_t diff = pVect1[i] - pVect2[i];
                dist += diff * diff;
            }

            return dist;
        }

// K is the unroll factor
        template <typename dist_t, typename data_t, int dimension = INT32_MAX>
        __force_inline__ float L2Distance(const float *pVect1, const float *pVect2,
                                          int dim) {

            static_assert(std::is_same<dist_t, float>::value,
                          "Argument dist_t must be of type float");
            static_assert(std::is_same<data_t, float>::value,
                          "Argument data_t must be of type float");

#ifdef __AVX512F__
            {
                __m512 temp = _mm512_set1_ps(0);
                constexpr int scale = sizeof(temp) / sizeof(float);

                float __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
                for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
                    const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(pVect1 + i),
                                                      _mm512_loadu_ps(pVect2 + i));
                    temp = _mm512_add_ps(temp, _mm512_mul_ps(diff, diff));
                }
                _mm512_store_ps(TmpRes, temp);
                return Sum16(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                        pVect1, pVect2, dim);
                ;
            }
#elif __AVX__
            {
    __m256 temp = _mm256_set1_ps(0);
    constexpr int scale = sizeof(temp) / sizeof(float);

    float __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
    for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
      const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(pVect1 + i),
                                        _mm256_loadu_ps(pVect2 + i));
      temp = _mm256_add_ps(temp, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, temp);
    return Sum8(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                              pVect1, pVect2, dim);
    ;
  }
#else
  {
    constexpr int scale = 1;
    return L2DistanceResidual<dist_t, data_t, scale, dimension>(pVect1, pVect2,
                                                                dim);
  }
#endif
        }

// K is the unroll factor
        template <typename dist_t, typename data_t, int dimension = INT32_MAX>
        __force_inline__ float L2Distance(const float16 *pVect1, const float16 *pVect2,
                                          int dim) {
            static_assert(std::is_same<dist_t, float>::value,
                          "Argument dist_t must be of type float");
            static_assert(std::is_same<data_t, float16>::value,
                          "Argument data_t must be of type float16");
#ifdef __AVX512F__
            {
                __m512 temp = _mm512_set1_ps(0);
                constexpr int scale = sizeof(temp) / sizeof(float);

                float __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
                for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
                    const __m512 diff =
                            _mm512_sub_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i_u *) (reinterpret_cast<const uint16_t *>(pVect1) + i))),
                                          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i_u *) (reinterpret_cast<const uint16_t *>(pVect2) + i))));
                    temp = _mm512_add_ps(temp, _mm512_mul_ps(diff, diff));
                }
                _mm512_store_ps(TmpRes, temp);
                return Sum16(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                        pVect1, pVect2, dim);
                ;
            }
#elif __AVX__ && __F16C__
            {
    __m256 temp = _mm256_set1_ps(0);
    constexpr int scale = sizeof(temp) / sizeof(float);

    float __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
    for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
      const __m256 diff = _mm256_sub_ps(
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(pVect1 + i))),
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(pVect2 + i))));
      temp = _mm256_add_ps(temp, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, temp);
    return Sum8(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                              pVect1, pVect2, dim);
    ;
  }
#else
  {
    constexpr int scale = 1;
    return L2DistanceResidual<dist_t, data_t, scale, dimension>(pVect1, pVect2,
                                                                dim);
  }
#endif
        }

// K is the unroll factor
        template <typename dist_t, typename data_t, int dimension = INT32_MAX>
        __force_inline__ int L2Distance(const uint8_t *pVect1, const uint8_t *pVect2,
                                        int dim) {
            static_assert(std::is_same<dist_t, int>::value,
                          "Argument dist_t must be of type int");
            static_assert(std::is_same<data_t, uint8_t>::value,
                          "Argument data_t must be of type uint8_t");
#ifdef __AVX512F__
            {
                __m512i temp = _mm512_set1_epi32(0);
                constexpr int scale = sizeof(temp) / sizeof(int16_t);

                int __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
                for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
                    const auto a_casted =
                            _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *)(pVect1 + i)));
                    const auto b_casted =
                            _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *)(pVect2 + i)));
                    const auto diff = _mm512_sub_epi16(a_casted, b_casted);
                    const auto diff_sq = _mm512_mullo_epi16(diff, diff);
                    temp = _mm512_add_epi32(
                            temp, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(diff_sq)));
                    temp = _mm512_add_epi32(
                            temp, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
                }
                _mm512_store_si512((__m512i *)TmpRes, temp);
                return Sum16(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                        pVect1, pVect2, dim);
                ;
            }
#elif __AVX__
            {
    __m256i temp = _mm256_set1_epi32(0);
    constexpr int scale = sizeof(temp) / sizeof(int16_t);

    int __attribute__((aligned(sizeof(temp)))) TmpRes[scale];

#pragma unroll
    for (int i = 0; i < get_main<scale, dimension>(dim); i += scale) {
      const auto a_casted = _mm256_cvtepu8_epi16(_mm_loadu_si128(
          (__m128i_u *)(pVect1 +
                        i))); // load 16 8-bit ints, unpack them two 16-bit int
      const auto b_casted =
          _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i_u *)(pVect2 + i)));
      const auto diff = _mm256_sub_epi16(
          a_casted, b_casted); // perform subtraction using 16 bit ints
      const auto diff_sq = _mm256_mullo_epi16(
          diff, diff); // perform multiplication using 16 bit ints
      temp = _mm256_add_epi32(
          temp, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(diff_sq)));
      temp = _mm256_add_epi32(
          temp, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_sq, 1)));
    }
    _mm256_store_si256((__m256i *)TmpRes, temp);
    return Sum8(TmpRes) + L2DistanceResidual<dist_t, data_t, scale, dimension>(
                              pVect1, pVect2, dim);
    ;
  }
#else
  {
    constexpr int scale = 1;
    return L2DistanceResidual<dist_t, data_t, scale, dimension>(pVect1, pVect2,
                                                                dim);
  }
#endif
        }
    } // namespace impl

// data_t will be cast to cast_t in register to conduct the computation by
// default unless the hardware supports the computation in cast_t natively
/**
 * @brief Wrapper for inner product with unroll optimization
 *
 * @tparam dist_t
 * @tparam data_t
 * @tparam cast_t
 * @param pVect1
 * @param pVect2
 * @param dim
 * @return dist_t
 */
    template <typename dist_t, typename data_t, typename cast_t>
    __force_inline__ dist_t L2Distance(const data_t *pVect1, const data_t *pVect2,
                                       int dim) {
        // optimize for specific dimension
        switch (dim) {
            case 200:
                return impl::L2Distance<dist_t, data_t, 200>(pVect1, pVect2, dim);
            case 128:
                return impl::L2Distance<dist_t, data_t, 128>(pVect1, pVect2, dim);
            case 100:
                return impl::L2Distance<dist_t, data_t, 100>(pVect1, pVect2, dim);
            case 96:
                return impl::L2Distance<dist_t, data_t, 96>(pVect1, pVect2, dim);
            case 64:
                return impl::L2Distance<dist_t, data_t, 96>(pVect1, pVect2, dim);
            case 32:
                return impl::L2Distance<dist_t, data_t, 96>(pVect1, pVect2, dim);
            case 16:
                return impl::L2Distance<dist_t, data_t, 16>(pVect1, pVect2, dim);
            default:
                return impl::L2Distance<dist_t, data_t>(pVect1, pVect2, dim);
                ;
        }
    };

    __force_inline__ float L2Distance(const float *pVect1, const float *pVect2, int dim) {
        return L2Distance<float, float, float>(pVect1, pVect2, dim);
    };

    __force_inline__ float L2Distance(const float16 *pVect1, const float16 *pVect2, int dim) {
        return L2Distance<float, float16, float>(pVect1, pVect2, dim);
    };

    __force_inline__ int L2Distance(const uint8_t *pVect1, const uint8_t *pVect2, int dim) {
        return L2Distance<int, uint8_t, int16_t>(pVect1, pVect2, dim);
    };
} // namespace melee
#endif //MELEE_L2_HPP
