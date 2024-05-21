#pragma once
#include "hnswlib.h"
#include <limits.h>
#include <cstdint>

namespace hnswlib {

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

//static int
//L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
//    size_t qty = *((size_t *) qty_ptr);
//    int res = 0;
//    unsigned char *a = (unsigned char *) pVect1;
//    unsigned char *b = (unsigned char *) pVect2;
//
//    qty = qty >> 2;
//    for (size_t i = 0; i < qty; i++) {
//        res += ((*a) - (*b)) * ((*a) - (*b));
//        a++;
//        b++;
//        res += ((*a) - (*b)) * ((*a) - (*b));
//        a++;
//        b++;
//        res += ((*a) - (*b)) * ((*a) - (*b));
//        a++;
//        b++;
//        res += ((*a) - (*b)) * ((*a) - (*b));
//        a++;
//        b++;
//    }
//    return (res);
//}
//
//static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
//    size_t qty = *((size_t*)qty_ptr);
//    int res = 0;
//    unsigned char* a = (unsigned char*)pVect1;
//    unsigned char* b = (unsigned char*)pVect2;
//
//    for (size_t i = 0; i < qty; i++) {
//        res += ((*a) - (*b)) * ((*a) - (*b));
//        a++;
//        b++;
//    }
//    return (res);
//}
//
//class L2SpaceI : public SpaceInterface<int> {
//    DISTFUNC<int> fstdistfunc_;
//    size_t data_size_;
//    size_t dim_;
//
// public:
//    L2SpaceI(size_t dim) {
//        if (dim % 4 == 0) {
//            fstdistfunc_ = L2SqrI4x;
//        } else {
//            fstdistfunc_ = L2SqrI;
//        }
//        dim_ = dim;
//        data_size_ = dim * sizeof(unsigned char);
//    }
//
//    size_t get_data_size() {
//        return data_size_;
//    }
//
//    DISTFUNC<int> get_dist_func() {
//        return fstdistfunc_;
//    }
//
//    void *get_dist_func_param() {
//        return &dim_;
//    }
//
//    ~L2SpaceI() {}
//};

 #define Sum8(arr)                                                              \
   arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]

 #define Sum16(arr)                                                             \
   arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] +      \
       arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] +      \
       arr[15]

 template <int scale, int dimension> constexpr int get_main(int dim) {
   if constexpr (dimension == INT_MAX) {
     return dim - dim % scale;
   } else {
     return dimension - dimension % scale;
   }
 };

 template <int scale, int dimension> constexpr int get_residual(int dim) {
   if constexpr (dimension == INT_MAX) {
     return dim % scale;
   } else {
     return dimension % scale;
   }
 };

 template <int scale, int dimension> constexpr int get_all(int dim) {
   if constexpr (dimension == INT_MAX) {
     return dim;
   } else {
     return dimension;
   }
 };

 template <typename dist_t, typename data_t, int scale,
           int dimension = INT32_MAX>
 inline dist_t L2DistanceResidual(const data_t *pVect1,
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
 inline int L2Distance(const uint8_t *pVect1, const uint8_t *pVect2,
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
 inline dist_t L2Distance(const data_t *pVect1, const data_t *pVect2,
                                    int dim) {
   // optimize for specific dimension
   switch (dim) {
   case 200:
     return L2Distance<dist_t, data_t, 200>(pVect1, pVect2, dim);
   case 128:
     return L2Distance<dist_t, data_t, 128>(pVect1, pVect2, dim);
   case 100:
     return L2Distance<dist_t, data_t, 100>(pVect1, pVect2, dim);
   case 96:
     return L2Distance<dist_t, data_t, 96>(pVect1, pVect2, dim);
   case 64:
     return L2Distance<dist_t, data_t, 64>(pVect1, pVect2, dim);
   case 32:
     return L2Distance<dist_t, data_t, 32>(pVect1, pVect2, dim);
   case 16:
     return L2Distance<dist_t, data_t, 16>(pVect1, pVect2, dim);
   default:
     return L2Distance<dist_t, data_t>(pVect1, pVect2, dim);
     ;
   }
 };

 int L2DistanceUint8(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
   return L2Distance<int, uint8_t, int16_t>((const uint8_t*) pVect1, (const uint8_t*) pVect2, *(size_t *)(qty_ptr));
 };

 class L2SpaceI : public SpaceInterface<int> {
     DISTFUNC<int> fstdistfunc_;
     size_t data_size_;
     size_t dim_;

  public:
     L2SpaceI(size_t dim) {
         fstdistfunc_ = L2DistanceUint8;
         dim_ = dim;
         data_size_ = dim * sizeof(unsigned char);
     }

     size_t get_data_size() {
         return data_size_;
     }

     DISTFUNC<int> get_dist_func() {
         return fstdistfunc_;
     }

     void *get_dist_func_param() {
         return &dim_;
     }

     ~L2SpaceI() {}
 };
}  // namespace hnswlib
