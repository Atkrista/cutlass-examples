#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "helpers.hpp"

template <class ProblemShape, class CtaTiler, Class TA, class AStride, class ASmemLayout, class AThreadLayout, class TB, class BStride, class BSmemLayout, class BThreadLayout, class TC, class CStride, class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK, Ctatiler cta_tiler,
                                                                                             TA const *A, Astride dA, ASmemLayout sA_layout, AthreadLayout tA,
                                                                                             TB const *B, Bstride dB, BSmemLayout sB_layout, BthreadLayout tB,
                                                                                             TC *C, Cstride dC, CSmemLayout, CthreadLayout tC, Alpha alpha, Beta beta)
{
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});            // (M, N, K)
    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

    // Static smem layouts
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    // Verify smem shapes with cta_tiler
    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

    // Thread partitioning
    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);
    static_assert(is_static<CThreadLayout>::value);

    // NumThreads
    CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
    CUTE_STATIC_ASSERT_V(size(tC) == size(tA));

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{}); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{}); // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{}); // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{}); // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{}); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{}); // BLK_N / THR_N

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

    // Get appropriate tile for this CTA
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    /**
     * Tensor gA_mk = zipped_divide(mA, select<0,2>(cta_tiler));
     * Tensor gA = gA_mk(make_coord(_,_), select<0,2>(cta_coord));
     */
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ TA smemA[cosize_v<sA_layout>];
    __shared__ TB smemB[cosize_v<sB_layout>];

    // Shared memory tensors
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)

    // Use a single thread to copy.
    // if (thread0())
    // {
    //     Tensor gA0 = gA(_, _, 0);
    //     for (int i = 0; i < size(sA); ++i)
    //     {
    //         sA(i) = gA0(i);
    //     }
    // }

    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

    // Partition sA (M,K) by the rows of tC
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
    // Partition sB (N,K) by the cols of tC
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    // Allocate the accumulators -- same shape/layout as the partitioned data
    Tensor tCrC = make_tensor_like(tCgC); // (THR_M,THR_N)

    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC)); // THR_M
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB)); // BLK_K

    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K

    // TUTORIAL: Example of a very simple compute mainloop
    //   copy(.) operates on the global and shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared and register memory via the tC partitioning

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Copy gmem to smem with tA|tB thread-partitioned tensors
        copy(tAgA(_, _, k_tile), tAsA); // A   (THR_M,THR_K) -> (THR_M,THR_K)
        copy(tBgB(_, _, k_tile), tBsB); // B   (THR_N,THR_K) -> (THR_N,THR_K)

        cp_async_fence();   // Label the end of (potential) cp.async instructions
        cp_async_wait<0>(); // Sync on all (potential) cp.async instructions
        __syncthreads();    // Wait for all threads to write to smem

        // Compute gemm on tC thread-partitioned smem
        gemm(tCsA, tCsB, tCrC); // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
        __syncthreads();        // Wait for all threads to read from smem
    }
}

int main(int argc, char **argv)
{
    const int s = 4096;
    int m = s;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = s;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = s;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    int p = s;
    if (argc >= 4)
        sscanf(argv[4], "%d", &p);

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto P = int(p);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TD = cute::half_t;
    using TE = cute::half_t;
    using TI = cute::half_t;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    // Allocate and Initialize host vectors
    thrust::host_vector<TA> h_A(M * K);
    thrust::host_vector<TB> h_B(N * K);
    thrust::host_vector<TC> h_C(M * N, TC(0.0f));
    thrust::host_vector<TD> h_D(P * N);
    thrust::host_vector<TE> h_E(M * P, TE(0.0f));
    random_fill<thrust::host_vector<TA>, TA>(h_A);
    random_fill<thrust::host_vector<TB>, TB>(h_B);
    random_fill<thrust::host_vector<TD>, TD>(h_D);

    // Allocate and init device vectors
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TD> d_D = h_D;
    thrust::device_vector<TE> d_E = h_E;

    // NT kernel
    int ldA = M;
    int ldB = N;
    int ldC = M;
    int ldD = P;

    // GEMM_1 = (m,n,k)
    // C = alpha * AB + beta * C
    auto prob_shape_prod = make_shape(M, N, K);
    // GEMM_2 = (m,p,n)
    // E = alpha * CD + beta * E
    auto prob_shape_cons = make_shape(M, P, N);

    // Define strides (static + dynamic)
    // Major in Mode-1 (K-major)
    // auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
    // auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
    // auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
    // auto dD = make_stride(ldD, Int<1>{}); // (dP, dN)
    // Major in Mode-0 (M-major)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)
    auto dD = make_stride(Int<1>{}, ldD); // (dP, dN)

    // Define CTA tile sizes (static)
    // Use the same tiler for prod & cons GEMMs
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tile = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)

    // Define Smem layouts
    // Prod
    auto sA = make_layouts(make_shape(bM, bK)); // (m, k) -> smem_index; m-major
    auto sB = make_layouts(make_shape(bM, bK));
    // Cons
    auto sC = make_layouts(make_shape(bM, bK));
    auto sD = make_layouts(make_shape(bP, bK));

    // Define thread layouts (static)
    // Partition each (BLK_M, BLK_K) and (BLK_N, BLK_K) tile among threads
    // Use same thread layout for prod & cons GEMMs
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (m,k) -> thr_idx; m-major
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (n,k) -> thr_idx; n-major
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // (m,n) -> thr_idx; m-major
}
