#include <functional>
#ifndef CODECUDA_CU
#define CODECUDA_CU

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include "CodeCommon.hpp"
#include "CodeInclude.h"
#include "cublas.h"
#include "cuda_runtime.h"


namespace code_kernels
{
    // A: M×K, B: K×N, C: M×N  (standard BLAS - K is the shared/inner dimension)
    // -----------------------------------------------------------------------------
    // 1D naive kernel: one CUDA thread computes one C(row, col).
    __global__ void k_matmul(const int M, const int N, const int K, float alpha, float beta, const float *a,
                             const float *b, float *c)
    {
        uint32_t x_global = ((blockDim.x * blockIdx.x) + threadIdx.x);
        auto y = static_cast<uint32_t>(x_global / N);
        if (y >= M)
        {
            return;
        }
        uint32_t x = x_global % N;
        float acc = 0;
        for (int i = 0; i < K; ++i)
        {
            acc += a[K * y + i] * b[N * i + x];
        }

        c[y * N + x] = alpha * acc + 1.0f * beta;
    }

    // -----------------------------------------------------------------------------
    // 2D naive kernel: grid.x maps N columns, grid.y maps M rows.
    __global__ void k_matmul_naive(const int M, const int N, const int K, float alpha, float beta, const float *a,
                                   const float *b, float *c)
    {
        uint32_t x = ((blockDim.x * blockIdx.x) + threadIdx.x);
        uint32_t y = ((blockDim.y * blockIdx.y) + threadIdx.y);

        if (y < M && x < N)
        {
            float acc = 0;
            for (int i = 0; i < K; ++i)
            {
                acc += a[y * K + i] * b[N * i + x];
            }

            c[y * N + x] = acc;
        }
    }

    // -----------------------------------------------------------------------------
    // Shared-memory tiled kernel: one block computes a 32x32 tile of C.
    __global__ void k_matmul_x_y(const int M, const int N, const int K, float alpha, float beta, const float *a,
                                 const float *b, float *c)
    {
        constexpr int32_t BLOCKSIZE = 32;
        // blockIdx.x maps M rows in these kernels.
        extern __shared__ float smem[];
        float *a_s = smem;
        float *b_s = smem + BLOCKSIZE * BLOCKSIZE;

        uint32_t row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
        uint32_t col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

        uint32_t local_row = (threadIdx.x / BLOCKSIZE);
        uint32_t local_col = (threadIdx.x % BLOCKSIZE);

        a += blockIdx.x * BLOCKSIZE * K;
        b += blockIdx.y * BLOCKSIZE;

        int tile_count = ceilf(float(K) / 32.0f);

        float tmp = 0.0f;
        for (int i = 0; i < tile_count; ++i)
        {
            bool a_bounds = (row < M) && (i * BLOCKSIZE + local_col < K);
            bool b_bounds = (i * BLOCKSIZE + local_row < K) && (col < N);

            a_s[local_row * BLOCKSIZE + local_col] = a_bounds ? a[local_row * K + local_col] : 0.0f;
            b_s[local_row * BLOCKSIZE + local_col] = b_bounds ? b[local_row * N + local_col] : 0.0f;

            __syncthreads();

            a += BLOCKSIZE;
            b += BLOCKSIZE * N;
#pragma unroll
            for (int j = 0; j < BLOCKSIZE; ++j)
            {
                tmp += a_s[local_row * BLOCKSIZE + j] * b_s[j * BLOCKSIZE + local_col];
            }
            __syncthreads();
        }
        if (row < M && col < N)
        {
            c[row * N + col] = tmp;
        }
    }

    // -----------------------------------------------------------------------------
    // Block-tiled row kernel: each thread accumulates BK values along N.
    __global__ void k_matmul_bt_row_tile(const int M, const int N, const int K, float alpha, float beta, const float *a,
                                         const float *b, float *c)
    {

        constexpr int BM = 64;
        constexpr int BK = 8;
        constexpr int BN = 64;
        // blockIdx.x maps M rows in these kernels.
        extern __shared__ float smem[];

        float *a_s = smem;
        float *b_s = smem + (BM * BK);

        uint32_t row = blockIdx.x * BM + (threadIdx.x / (BK));
        uint32_t local_col = (threadIdx.x % (BK));

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);

        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);
        int b_stride = BN / BK;

        a += blockIdx.x * BM * K;
        b += blockIdx.y * b_stride;

        float bt[BK] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        // k
        for (int i = 0; i < tile_count; ++i)
        {
            a_s[local_row_a * BK + local_col_a] = a[local_row_a * K + local_col_a];
            b_s[local_row_b * BN + local_col_b] = b[local_row_b * N + local_col_b];
            __syncthreads();

            a += BK;
            b += BK * N;
            for (int j = 0; j < BK; ++j)
            {
                float a_temp = a_s[local_row_a * BK + j];
                for (int k = 0; k < BK; ++k)
                {
                    bt[k] += a_temp * b_s[j * BN + local_col];
                    //(a0,0 * b0,0)k++... (a0,0 * b8,0...)k++... (a0,0 * b16,0)k++ (j++)
                    //(a1,0 * b0,1)k++... (a1,0 * b8,1...)k++... (a1,0 * b16,1)k++ (j++)
                    //(a2,0 * b0,2)k++... (a2,0 * b8,2...)k++... (a2,0 * b16,2)k++ (j++)
                }
            }
            __syncthreads();
        }

        if (row < M)
        {
        }
        int total_c_entries_per_block = (blockIdx.y * BK * BK);
        // each thread cal 0 ... 8 ... 16 ... 24 ... 32 ... etc
        for (int col_offset = 0; col_offset < BK; ++col_offset)
        {
            c[(row * N) + (BK * col_offset) + total_c_entries_per_block + local_col] = bt[col_offset];
        }
    }

    // -----------------------------------------------------------------------------
    // Block-tiled column kernel: each thread accumulates BK values along M.
    __global__ void k_matmul_bt_col_tile(const int M, const int N, const int K, float alpha, float beta, const float *a,
                                         const float *b, float *c)
    {

        constexpr int BM = 64;
        constexpr int BK = 8;
        constexpr int BN = 64;
        // blockIdx.x maps M rows in these kernels.
        // y = rows
        extern __shared__ float smem[];

        float *a_s = smem;
        float *b_s = smem + (BM * BK);

        uint32_t row_thread = threadIdx.x / BN;
        uint32_t col_thread = threadIdx.x % BN;

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);

        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);

        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        c += (blockIdx.y * BM * N) + (blockIdx.x * BN);

        float bt[BK] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        // k
        for (int i = 0; i < tile_count; ++i)
        {
            a_s[local_row_a * BK + local_col_a] = a[local_row_a * K + local_col_a];
            b_s[local_row_b * BN + local_col_b] = b[local_row_b * N + local_col_b];
            __syncthreads();

            a += BK;
            b += BK * N;
            for (int j = 0; j < BK; ++j)
            {
                float b_temp = b_s[j * BN + col_thread];
                for (int k = 0; k < BK; ++k)
                {
                    bt[k] += a_s[(BK * BK * row_thread) + (BK * k) + j] * b_temp;
                }
            }
            __syncthreads();
        }
        for (int row_offset = 0; row_offset < BK; ++row_offset)
        {
            c[(row_thread * BK * N) + (row_offset * N) + col_thread] = bt[row_offset];
        }
    }

    // -----------------------------------------------------------------------------
    // 2D register-tiled kernel: each thread computes an TMxTN output tile.
    __global__ void k_matmul_bt_2d_tilling(const int M, const int N, const int K, float alpha, float beta,
                                           const float *a, const float *b, float *c)
    {
        // blockIdx.x maps M rows in these kernels.
        // y = rows
        constexpr uint32_t BM = 64;
        constexpr uint32_t BK = 8;
        constexpr uint32_t BN = 64;
        constexpr uint32_t TM = 8;
        constexpr uint32_t TN = 8;

        extern __shared__ float smem[];

        float *a_s = smem;
        float *b_s = smem + (BM * BK);

        uint32_t row_thread = threadIdx.x / BK;
        uint32_t col_thread = threadIdx.x % BK;

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);

        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);

        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        c += (blockIdx.y * BM * N) + (blockIdx.x * BN);

        float bt[TM * TN] = {0.0};
        float reg_m[TM] = {0.0};
        float reg_n[TN] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));
        // CODECUDA_DEVICE_LOG_INFO("%d", tile_count);

        // keep in mind for this matmul algorithms there is a lot of sizes that need to match
        //  in this case is not coincidence that BK * BN/BK * BK = 8, that means we can safely jump on
        // a and b by BM * K and in B by BN
        for (int i = 0; i < tile_count; ++i)
        {
            for (uint32_t curr_stride = 0; curr_stride < BM; curr_stride += 8)
            {
                uint32_t stride_row_offset = local_row_a * BK + curr_stride * BK;
                a_s[stride_row_offset + local_col_a] = a[(local_row_a + curr_stride) * K + local_col_a];
            }
            for (uint32_t curr_stride = 0; curr_stride < BK; ++curr_stride)
            {
                uint32_t stride_row_offset = curr_stride * BN;
                b_s[stride_row_offset + local_col_b] = b[curr_stride * N + local_col_b];
            }
            __syncthreads();

            a += BK;
            b += BK * N;
            for (int dot_idx = 0; dot_idx < BK; ++dot_idx)
            {
                for (int reg_row_idx = 0; reg_row_idx < TM; ++reg_row_idx)
                {
                    reg_m[reg_row_idx] = a_s[row_thread * TM * TM + reg_row_idx * TM + dot_idx];
                }
                for (int reg_col_idx = 0; reg_col_idx < TN; ++reg_col_idx)
                {
                    reg_n[reg_col_idx] = b_s[dot_idx * BN + col_thread * TN + reg_col_idx];
                }
                for (int reg_n_idx = 0; reg_n_idx < TN; reg_n_idx++)
                {
                    for (int reg_m_idx = 0; reg_m_idx < TM; ++reg_m_idx)
                    {
                        bt[reg_m_idx * TM + reg_n_idx] += reg_m[reg_m_idx] * reg_n[reg_n_idx];
                    }
                }
            }
            __syncthreads();
        }
        for (int row_offset = 0; row_offset < BK; ++row_offset)
        {
            for (int col_offset = 0; col_offset < BK; ++col_offset)
            {
                c[(row_thread * BK + row_offset) * N + (col_thread * BK) + col_offset] =
                    bt[row_offset * BK + col_offset];
            }
        }
    }

    // -----------------------------------------------------------------------------
    // 2D register-tiled kernel with A transposed in shared memory.
    __global__ void k_matmul_bt_2d_tilling_transposed_a(const int M, const int N, const int K, float alpha, float beta,
                                                        float *a, float *b, float *c)
    {
        // blockIdx.x maps M rows in these kernels.
        // y = rows
        constexpr uint32_t BM = 64;
        constexpr uint32_t BK = 8;
        constexpr uint32_t BN = 64;
        constexpr uint32_t TM = 8;
        constexpr uint32_t TN = 8;

        extern __shared__ float smem[];

        float *a_s = smem;
        float *b_s = smem + (BM * BK);

        uint32_t row_thread = threadIdx.x / BK;
        uint32_t col_thread = threadIdx.x % BK;

        uint32_t local_col_b = (threadIdx.x % BN);

        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        c += (blockIdx.y * BM * N) + (blockIdx.x * BN);

        float bt[TM * TN] = {0.0};
        float reg_m[TM] = {0.0};
        float reg_n[TN] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        // keep in mind for this matmul algorithms there is a lot of sizes that need to match
        //  in this case is not coincidence that BK * BN/BK * BK = 8, that means we can safely jump on
        // a and b by BM * K and in B by BN
        for (int i = 0; i < tile_count; ++i)
        {

            float4 *a_buff = reinterpret_cast<float4 *>(&a[(threadIdx.x) * K]);
            float4 temp_a_1 = a_buff[0];
            float4 temp_a_2 = a_buff[1];

            // transpose A for vectorized loads on the outer product
            a_s[0 * BM + threadIdx.x] = temp_a_1.x;
            a_s[1 * BM + threadIdx.x] = temp_a_1.y;
            a_s[2 * BM + threadIdx.x] = temp_a_1.z;
            a_s[3 * BM + threadIdx.x] = temp_a_1.w;
            a_s[4 * BM + threadIdx.x] = temp_a_2.x;
            a_s[5 * BM + threadIdx.x] = temp_a_2.y;
            a_s[6 * BM + threadIdx.x] = temp_a_2.z;
            a_s[7 * BM + threadIdx.x] = temp_a_2.w;
            for (uint32_t curr_stride = 0; curr_stride < BK; ++curr_stride)
            {
                uint32_t stride_row_offset = curr_stride * BN;
                b_s[stride_row_offset + local_col_b] = b[curr_stride * N + local_col_b];
            }
            __syncthreads();

            a += BK;
            b += BK * N;
            for (int dot_idx = 0; dot_idx < BK; ++dot_idx)
            {
                for (int reg_col_idx = 0; reg_col_idx < TM; ++reg_col_idx)
                {
                    reg_m[reg_col_idx] = a_s[dot_idx * BM + row_thread * TM + reg_col_idx];
                }
                for (int reg_col_idx = 0; reg_col_idx < TN; ++reg_col_idx)
                {
                    reg_n[reg_col_idx] = b_s[dot_idx * BN + col_thread * TN + reg_col_idx];
                }
                for (int reg_n_idx = 0; reg_n_idx < TN; reg_n_idx++)
                {
                    for (int reg_m_idx = 0; reg_m_idx < TM; ++reg_m_idx)
                    {
                        bt[reg_m_idx * TM + reg_n_idx] += reg_m[reg_m_idx] * reg_n[reg_n_idx];
                    }
                }
            }
            __syncthreads();
        }
        for (int row_offset = 0; row_offset < BK; ++row_offset)
        {
            for (int col_offset = 0; col_offset < BK; ++col_offset)
            {
                c[(row_thread * BK + row_offset) * N + (col_thread * BK) + col_offset] =
                    bt[row_offset * BK + col_offset];
            }
        }
    }
    /*
        We have this levels
        - WM/WN = How much of BM and BN we are going to calculate per warp
        The output is going to be WM * WN per warp
        - TM/TN = How many cols and rows from M and cols from N we calculate
        - WSUBM/WSUBN is the size of times we divide WM/WN based on TM/TN
        - WMITER/WNITER number of iterations we do based on
        TM/TN tiles we calculate per thread

        // warp size
        WSIZE = 32
        //block size on M
        BM = 64
        //block size on N
        WN = 128
        //warp tile on M
        WM = 32
        //warp tile on N
        WN = 64
        //number of subtiles on the warptile in WN
        WNITER = 2
        //number of subtiles on the warptile in WM

        //number of entries in N used per thread
        TN = 4
        TM = 4

        //Number of subtiles across WM
        WMITER = (WM * WN)/(WSIZE * TM * TN * WNITER)
        //size of the warpsubtile in WN
        WSUBN = WN / WNITER = 64 / 2 = 32

        //size of the warpsubtile in WM
        WSUBM = WM / WMITER = 32 / 2 = 16

        //number of cols the warp will have
        WTCOLS = WSUBN / TN = 32 / 4 = 8
        //number of rows the warp will have
        WTROWS = WSIZE / WTCOLS = 32 / 8 = 4

        //number of entries in M used per thread
        TM = WSUBM/WTHREADROWS = 16 / 4 = 4

        //number of entries calculated in total per thread (TM * TN tiles of WMITER * WNITER)
        TRES = TM * WMITER * TN * WNITER = 4 * 2 * 4 * 2 = 64
        //number of register loaded in M
        REGM = TM * WMITER = 8
        //number of register loaded in N
        REGN = TN * WNITER = 8
    */
    struct k_auto_tunning_params
    {
        static constexpr uint32_t WSIZE = 32;
        static constexpr uint32_t BN = 128;
        static constexpr uint32_t BM = 64;
        static constexpr uint32_t BK = 16;
        static constexpr uint32_t WN = 64;
        static constexpr uint32_t WM = 32;
        // this is the total block size calculated based on BM, WM... so
        static constexpr uint32_t BSIZE = (BM / WM) * (BN / WN) * WSIZE;
        static constexpr uint32_t WCOLS = BN / WN;
        static constexpr uint32_t WROWS = BM / WM;
        static constexpr uint32_t WNITER = 2;

        static constexpr uint32_t TN = 4;
        static constexpr uint32_t TM = 4;

        static constexpr uint32_t WMITER = (WM * WN) / (WSIZE * TM * TN * WNITER);
        static constexpr uint32_t WSUBN = WN / WNITER;
        static constexpr uint32_t WSUBM = WM / WMITER;
        static constexpr uint32_t WTCOLS = WSUBN / TN;
        static constexpr uint32_t WTROWS = WSIZE / WTCOLS;
    };
    __global__ void k_matmul_bt_warp_tilling(const int M, const int N, const int K, float alpha, float beta, float *a,
                                             float *b, float *c)
    {
        // x = rows
        // y = rows
        using Params = k_auto_tunning_params;

        constexpr uint32_t WSIZE = Params::WSIZE;
        constexpr uint32_t BSIZE = Params::BSIZE;
        constexpr uint32_t BN = Params::BN;
        constexpr uint32_t BM = Params::BM;
        constexpr uint32_t BK = Params::BK;
        constexpr uint32_t WN = Params::WN;
        constexpr uint32_t WM = Params::WM;
        constexpr uint32_t WCOLS = Params::WCOLS;
        constexpr uint32_t WROWS = Params::WROWS;
        constexpr uint32_t WNITER = Params::WNITER;

        constexpr uint32_t TN = Params::TN;
        constexpr uint32_t TM = Params::TM;

        constexpr uint32_t WMITER = Params::WMITER;
        constexpr uint32_t WSUBN = Params::WSUBN;
        constexpr uint32_t WSUBM = Params::WSUBM;
        constexpr uint32_t WTCOLS = Params::WTCOLS;
        constexpr uint32_t WTROWS = Params::WTROWS;

        // number of subtiles calculated per thread

        extern __shared__ float smem[];

        float *a_s = smem;
        float *b_s = smem + (BM * BK);

        uint32_t warp_id = threadIdx.x / WSIZE;
        uint32_t lane_id = threadIdx.x % WSIZE;

        uint32_t warp_row = warp_id / WCOLS;
        uint32_t warp_col = warp_id % WCOLS;

        uint32_t thread_row_in_warp = lane_id / WTCOLS;
        uint32_t thread_col_in_warp = lane_id % WTCOLS;

        // 8
        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        // we place c on the warp block
        c += (blockIdx.y * BM * N + blockIdx.x * BN) + (warp_row * WM * N + warp_col * WN);

        float thread_results[TM * TN * WMITER * WNITER] = {0.0f};
        float reg_m[TM * WMITER] = {0.0f};
        float reg_n[TN * WNITER] = {0.0f};

        uint32_t tile_count = ceilf(float(K) / float(BK));

        // We make tiles of 4 along the rows because we are doing vec4 loads
        constexpr uint32_t BM_COLS = BK / 4;
        constexpr uint32_t BM_ROW_STRIDE = BSIZE / BM_COLS;

        constexpr uint32_t BN_COLS = BN / 4;
        constexpr uint32_t BN_ROW_STRIDE = BSIZE / BN_COLS;

        uint32_t a_col = threadIdx.x % BM_COLS;
        uint32_t a_row = threadIdx.x / BM_COLS;


        uint32_t b_col = threadIdx.x % BN_COLS;
        uint32_t b_row = threadIdx.x / BN_COLS;


        for (int i = 0; i < tile_count; ++i)
        {
            // // transpose A for vectorized loads on the outer product
            //
            // // transpose A for vectorized loads on the outer product
            for (int curr_stride = 0; curr_stride < BM; curr_stride += BM_ROW_STRIDE)
            {
                float4 *a_buff = reinterpret_cast<float4 *>(&a[(a_row + curr_stride) * K + a_col * 4]);
                float4 temp_a_1 = a_buff[0];
                a_s[(a_col * 4 + 0) * BM + (a_row + curr_stride)] = temp_a_1.x;
                a_s[(a_col * 4 + 1) * BM + (a_row + curr_stride)] = temp_a_1.y;
                a_s[(a_col * 4 + 2) * BM + (a_row + curr_stride)] = temp_a_1.z;
                a_s[(a_col * 4 + 3) * BM + (a_row + curr_stride)] = temp_a_1.w;
            }

            for (uint32_t curr_stride = 0; curr_stride < BK; curr_stride += BN_ROW_STRIDE)
            {
                float4 *b_buff = reinterpret_cast<float4 *>(&b[(b_row + curr_stride) * N + b_col * 4]);
                float4 temp = b_buff[0];
                b_s[(b_row + curr_stride) * BN + (b_col * 4 + 0)] = temp.x;
                b_s[(b_row + curr_stride) * BN + (b_col * 4 + 1)] = temp.y;
                b_s[(b_row + curr_stride) * BN + (b_col * 4 + 2)] = temp.z;
                b_s[(b_row + curr_stride) * BN + (b_col * 4 + 3)] = temp.w;
            }
            __syncthreads();

            a += BK;
            b += BK * N;

            for (int dot_idx = 0; dot_idx < BK; ++dot_idx)
            {
                for (int wm_iter_idx = 0; wm_iter_idx < WMITER; ++wm_iter_idx)
                {
                    for (int tm_idx = 0; tm_idx < TM; ++tm_idx)
                    {
                        reg_m[TM * wm_iter_idx + tm_idx] =
                            a_s[(dot_idx * BM) + (warp_row * WM) + (WSUBM * wm_iter_idx) + (TM * thread_row_in_warp) +
                                tm_idx];
                    }
                }

                for (int wn_iter_idx = 0; wn_iter_idx < WNITER; ++wn_iter_idx)
                {
                    for (int tn_idx = 0; tn_idx < TN; ++tn_idx)
                    {
                        reg_n[TN * wn_iter_idx + tn_idx] =
                            b_s[(dot_idx * BN) + (warp_col * WN) + (WSUBN * wn_iter_idx) + (TN * thread_col_in_warp) +
                                tn_idx];
                    }
                }

                for (int wm_iter_idx = 0; wm_iter_idx < WMITER; ++wm_iter_idx)
                {
                    for (int wn_iter_idx = 0; wn_iter_idx < WNITER; ++wn_iter_idx)
                    {
                        for (int reg_n_idx = 0; reg_n_idx < TN; ++reg_n_idx)
                        {
                            for (int reg_m_idx = 0; reg_m_idx < TM; ++reg_m_idx)
                            {
                                thread_results[(WNITER * TM * TN * wm_iter_idx) + (WNITER * TN * reg_m_idx) +
                                               (TN * wn_iter_idx) + reg_n_idx] +=
                                    reg_m[TM * wm_iter_idx + reg_m_idx] * reg_n[TN * wn_iter_idx + reg_n_idx];
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }

        for (int wm_iter_idx = 0; wm_iter_idx < WMITER; ++wm_iter_idx)
        {
            for (int wn_iter_idx = 0; wn_iter_idx < WNITER; ++wn_iter_idx)
            {
                float *c_temp = c;
                c_temp += wm_iter_idx * WSUBM * N + wn_iter_idx * WSUBN;
                for (int tm_idx = 0; tm_idx < TM; ++tm_idx)
                {
                    for (int tn_idx = 0; tn_idx < TN; tn_idx += 4)
                    {
                        // vectorized loads for c+ tm_idx * N + tn_idx
                        float4 tmp = reinterpret_cast<float4 *>(
                            &c_temp[(thread_row_in_warp * TM * N) + (TN * thread_col_in_warp) + tm_idx * N])[0];

                        tmp.x = alpha *
                                thread_results[(WNITER * TM * TN * wm_iter_idx) + (WNITER * TM * tm_idx) +
                                               (TN * wn_iter_idx + 0)] +
                            tmp.x * beta;
                        tmp.y = alpha *
                                thread_results[(WNITER * TM * TN * wm_iter_idx) + (WNITER * TM * tm_idx) +
                                               (TN * wn_iter_idx + 1)] +
                            tmp.y * beta;
                        tmp.z = alpha *
                                thread_results[(WNITER * TM * TN * wm_iter_idx) + (WNITER * TM * tm_idx) +
                                               (TN * wn_iter_idx + 2)] +
                            tmp.z * beta;
                        tmp.w = alpha *
                                thread_results[(WNITER * TM * TN * wm_iter_idx) + (WNITER * TM * tm_idx) +
                                               (TN * wn_iter_idx + 3)] +
                            tmp.w * beta;
                        reinterpret_cast<float4 *>(
                            &c_temp[(thread_row_in_warp * TM * N) + (TN * thread_col_in_warp) + tm_idx * N])[0] = tmp;
                        // ;
                    }
                }
            }
        }
    }

    __global__ void k_check_mat_err(const int M, const int N, const float *a, const float *b, float *c)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N)
            return;
        c[idx] = fabsf(a[idx] - b[idx]);
    }
} // namespace code_kernels

namespace CodeCuda
{
    namespace Internals
    {
        struct kernel_launcher
        {
            std::function<void()> kernel;
        };

    } // namespace Internals
    namespace Wrappers
    {
        inline void C_Free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }
        inline void C_DeviceSynchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }
        inline void C_GetLastError() { CUDA_CHECK(cudaGetLastError()); }
        inline void C_FreeZero() { CUDA_CHECK(cudaFree(0)); }
        inline void C_StreamCreate(cudaStream_t *stream) { CUDA_CHECK(cudaStreamCreate(stream)); }
        inline void C_StreamDestroy(cudaStream_t stream) { CUDA_CHECK(cudaStreamDestroy(stream)); }
        inline void C_StreamSynchronize(cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
        inline void C_EventCreate(cudaEvent_t *event) { CUDA_CHECK(cudaEventCreate(event)); }
        inline void C_EventDestroy(cudaEvent_t event) { CUDA_CHECK(cudaEventDestroy(event)); }
        inline void C_EventRecord(cudaEvent_t event, cudaStream_t stream = 0)
        {
            CUDA_CHECK(cudaEventRecord(event, stream));
        }
        inline void C_EventSynchronize(cudaEvent_t event) { CUDA_CHECK(cudaEventSynchronize(event)); }
        inline void C_EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)
        {
            CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
        }
        inline void C_Memset(void *dst, int value, size_t count) { CUDA_CHECK(cudaMemset(dst, value, count)); }
        inline void C_SetDevice(int device) { CUDA_CHECK(cudaSetDevice(device)); }
        inline void C_GetDevice(int *device) { CUDA_CHECK(cudaGetDevice(device)); }
        template <class T>
        inline void C_HostMalloc(T **ptr, size_t size, unsigned int flags = cudaHostAllocDefault)
        {
            CUDA_CHECK(cudaHostAlloc((void **)ptr, size, flags));
        }
        inline void C_HostFree(void *ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
        template <class T>
        inline void C_MallocManaged(T **ptr, size_t size, unsigned int flags = cudaMemAttachGlobal)
        {
            CUDA_CHECK(cudaMallocManaged((void **)ptr, size, flags));
        }
        template <class T>
        inline void C_Malloc(T **ptr, size_t size)
        {
            CUDA_CHECK(cudaMalloc((void **)ptr, size));
        }
        inline void C_Memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
        {
            CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
        }
        inline void C_MemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, count, kind, stream));
        }
        inline void C_DeviceReset() { CUDA_CHECK(cudaDeviceReset()); }

    } // namespace Wrappers


    void C_Init()
    {
        cublasInit();
        CODECUDA_PRINTLN("Initialized codeCudaLib");
    }

    void C_Matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        if (c == nullptr)
        {
            CODECUDA_LOG_WARNING("target buffer is empty");
            return;
        }
        float *d_A, *d_B, *d_C;
        Wrappers::C_Malloc(&d_A, M * K * sizeof(float));
        Wrappers::C_Malloc(&d_B, K * N * sizeof(float));
        Wrappers::C_Malloc(&d_C, M * N * sizeof(float));

        Wrappers::C_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
        Wrappers::C_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        constexpr uint32_t BM = code_kernels::k_auto_tunning_params::BM;
        constexpr uint32_t BK = code_kernels::k_auto_tunning_params::BK;
        constexpr uint32_t BN = code_kernels::k_auto_tunning_params::BN;
        constexpr uint32_t BSIZE = code_kernels::k_auto_tunning_params::BSIZE;

        dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
        dim3 block(BSIZE);
        code_kernels::k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(
            M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        
        Wrappers::C_GetLastError();
        Wrappers::C_DeviceSynchronize();

        Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        Wrappers::C_Free(d_A);
        Wrappers::C_Free(d_B);
        Wrappers::C_Free(d_C);
    }

    void cpu_matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float temp = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    temp += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = temp;
            }
        }
    }

    namespace CodeBenchmarking
    {
        void add_kernel_launcher(const std::string &name, std::function<void()> kernelFunc,
                                 std::map<std::string, Internals::kernel_launcher> &kernels_out)
        {
            Internals::kernel_launcher launcher;
            launcher.kernel = std::move(kernelFunc);
            kernels_out.try_emplace(name, launcher);
        }

        std::string BuildMatmulBenchmarkResultJson(const c_matmul_benchmark_result &result)
        {
            using Params = code_kernels::k_auto_tunning_params;

            std::ostringstream output;
            output << std::boolalpha;
            output << "{\n";
            output << "  \"shape\": {\"M\": " << result.M << ", \"N\": " << result.N << ", \"K\": " << result.K
                   << "},\n";
            output << "  \"runs\": " << result.runs << ",\n";
            output << "  \"autotuning_params\": {\n";
            output << "    \"WSIZE\": " << Params::WSIZE << ",\n";
            output << "    \"BN\": " << Params::BN << ",\n";
            output << "    \"BM\": " << Params::BM << ",\n";
            output << "    \"BK\": " << Params::BK << ",\n";
            output << "    \"WN\": " << Params::WN << ",\n";
            output << "    \"WM\": " << Params::WM << ",\n";
            output << "    \"BSIZE\": " << Params::BSIZE << ",\n";
            output << "    \"WCOLS\": " << Params::WCOLS << ",\n";
            output << "    \"WROWS\": " << Params::WROWS << ",\n";
            output << "    \"WNITER\": " << Params::WNITER << ",\n";
            output << "    \"TN\": " << Params::TN << ",\n";
            output << "    \"TM\": " << Params::TM << ",\n";
            output << "    \"WMITER\": " << Params::WMITER << ",\n";
            output << "    \"WSUBN\": " << Params::WSUBN << ",\n";
            output << "    \"WSUBM\": " << Params::WSUBM << ",\n";
            output << "    \"WTCOLS\": " << Params::WTCOLS << ",\n";
            output << "    \"WTROWS\": " << Params::WTROWS << "\n";
            output << "  },\n";
            output << "  \"personal\": {\"kernel\": \"warp_tilling\", \"ms\": " << result.personal_ms
                   << ", \"gflops\": " << result.personal_gflops << "},\n";
            output << "  \"cublas\": {\"ms\": " << result.cublas_ms << ", \"gflops\": " << result.cublas_gflops
                   << "},\n";
            output << "  \"accuracy\": {\"average_error\": " << result.average_error
                   << ", \"max_error\": " << result.max_error << "},\n";
            output << "  \"passed\": " << result.passed << "\n";
            output << "}";
            return output.str();
        }

        std::string TrimTrailingWhitespace(std::string text)
        {
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
            {
                text.pop_back();
            }
            return text;
        }

        void C_SaveMatmulBenchmarkResultJson(const char *path, const c_matmul_benchmark_result &result)
        {
            if (path == nullptr || path[0] == '\0')
            {
                CODECUDA_LOG_WARNING("benchmark json path is empty");
                return;
            }

            const std::string result_json = BuildMatmulBenchmarkResultJson(result);

            std::ifstream existing_input(path);
            std::string existing;
            if (existing_input)
            {
                existing.assign(std::istreambuf_iterator<char>(existing_input), std::istreambuf_iterator<char>());
            }

            std::ofstream output(path, std::ios::out | std::ios::trunc);
            if (!output)
            {
                CODECUDA_LOG_WARNING("failed to open benchmark json path: ", path);
                return;
            }

            existing = TrimTrailingWhitespace(existing);
            if (existing.size() >= 2 && existing.front() == '[' && existing.back() == ']')
            {
                existing.pop_back();
                existing = TrimTrailingWhitespace(existing);
                output << existing;
                if (existing.size() > 1)
                {
                    output << ",\n";
                }
                output << result_json << "\n]\n";
                return;
            }

            output << "[\n" << result_json << "\n]\n";
        }

        void C_Matmul_Test(const int M, const int N, const int K, const float *a, const float *b, float *c, int runs)
        {
            if (c == nullptr)
            {
                CODECUDA_LOG_WARNING("target buffer is empty");
                return;
            }
            bool testPassed = true;

            // personal
            float *d_A, *d_B, *d_C;
            Wrappers::C_Malloc(&d_A, M * K * sizeof(float));
            Wrappers::C_Malloc(&d_B, K * N * sizeof(float));
            Wrappers::C_Malloc(&d_C, M * N * sizeof(float));

            Wrappers::C_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            Wrappers::C_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            Wrappers::C_EventCreate(&start);
            Wrappers::C_EventCreate(&stop);
            /*
            dim3 grid(ceil(double(N)/32.0f), ceil(double(M)/32.0f));
            dim3 block(32, 32);
            */

            std::map<std::string, Internals::kernel_launcher> kernels;

            add_kernel_launcher(
                "naive_coalescent",
                [N, M, K, d_A, d_B, d_C]()
                {
                    dim3 grid(ceil(double(N) / 32.0f), ceil(double(M) / 32.0f));
                    dim3 block(32, 32);
                    code_kernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);
            add_kernel_launcher(
                "smem",
                [N, M, K, d_A, d_B, d_C]()
                {
                    dim3 grid(ceil(double(M) / 32.0f), ceil(double(N) / 32.0f));
                    dim3 block(32 * 32);
                    code_kernels::k_matmul_x_y<<<grid, block, block.x * sizeof(float) * 2>>>(M, N, K, 1.0f, 0.0f, d_A,
                                                                                             d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_col_tile",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BM * BK);
                    code_kernels::k_matmul_bt_col_tile<<<grid, block, (BN * BK + BK * BM) * sizeof(float)>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_row_tile",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(M) / double(BM)), ceil(double(N) / double(BK * BK)));
                    dim3 block(BM * BK);
                    code_kernels::k_matmul_bt_row_tile<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_2d",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    code_kernels::k_matmul_bt_2d_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_2d_transposed",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    code_kernels::
                        k_matmul_bt_2d_tilling_transposed_a<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(
                            M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "warp_tilling",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = code_kernels::k_auto_tunning_params::BM;
                    constexpr uint32_t BK = code_kernels::k_auto_tunning_params::BK;
                    constexpr uint32_t BN = code_kernels::k_auto_tunning_params::BN;
                    constexpr uint32_t BSIZE = code_kernels::k_auto_tunning_params::BSIZE;

                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
                    dim3 block(BSIZE);
                    code_kernels::k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);


            for (int i = 0; i < 5; ++i)
            {
                kernels.at("warp_tilling").kernel();
            }
            Wrappers::C_GetLastError();
            Wrappers::C_DeviceSynchronize();

            Wrappers::C_EventRecord(start);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("warp_tilling").kernel();
            }
            Wrappers::C_GetLastError();

            Wrappers::C_EventRecord(stop);
            Wrappers::C_EventSynchronize(stop);
            float ms = 0;
            Wrappers::C_EventElapsedTime(&ms, start, stop);
            double ms_real = ms / double(runs);
            CODECUDA_PRINTLN("Average Time personal (ms): ", ms_real);
            auto flops = int64_t(2 * int64_t(M) * int64_t(N) * int64_t(K));

            double gflops = (double(flops) * 1.0e-9f) / double(ms_real / 1000.0f);
            CODECUDA_PRINTLN("GFLOPS/s personal: ", gflops);

            // cubulas
            float *d_A_cublas, *d_B_cublas, *d_C_cublas;
            Wrappers::C_Malloc(&d_A_cublas, M * K * sizeof(float));
            Wrappers::C_Malloc(&d_B_cublas, K * N * sizeof(float));
            Wrappers::C_Malloc(&d_C_cublas, M * N * sizeof(float));

            Wrappers::C_Memcpy(d_A_cublas, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            Wrappers::C_Memcpy(d_B_cublas, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            float alpha = 1.0f;
            float beta = 0.0f;
            cublasHandle_t handle;
            cudaEvent_t start_cublas, stop_cublas;
            Wrappers::C_EventCreate(&start_cublas);
            Wrappers::C_EventCreate(&stop_cublas);

            CUBLAS_CHECK(cublasCreate_v2(&handle));

            add_kernel_launcher(
                "cublas",
                [handle, N, M, K, alpha, d_B_cublas, d_A_cublas, beta, d_C_cublas]()
                {
                    CUBLAS_CHECK(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N,
                                                d_A_cublas, K, &beta, d_C_cublas, N));
                },
                kernels);

            for (int i = 0; i < 5; ++i)
            {
                kernels.at("cublas").kernel();
            }
            Wrappers::C_EventRecord(start_cublas);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("cublas").kernel();
            }
            Wrappers::C_EventRecord(stop_cublas);
            Wrappers::C_EventSynchronize(stop_cublas);
            float ms_cublas = 0;
            Wrappers::C_EventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
            double ms_real_cublas = ms_cublas / double(runs);
            CODECUDA_PRINTLN("Average Time cublas (ms): ", ms_real_cublas);
            double gflops_cublas = double(double(flops) * 1.0e-9) / double(ms_real_cublas / 1000.0);
            CODECUDA_PRINTLN("GFLOPS/s cublas: ", gflops_cublas);
            // personal vs cublas
            float *d_C_err;
            Wrappers::C_Malloc(&d_C_err, M * N * sizeof(float));

            dim3 block_err(128, 1, 1);
            dim3 grid_err(ceil(double(M * N) / 128.0), 1, 1);
            code_kernels::k_check_mat_err<<<grid_err, block_err>>>(M, N, d_C, d_C_cublas, d_C_err);
            Wrappers::C_GetLastError();
            Wrappers::C_DeviceSynchronize();

            float *errMatrix;
            errMatrix = new float[M * N];
            Wrappers::C_Memcpy(errMatrix, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            float err_average = 0.0f;
            float max_error = 0.0f;
            for (int i = 0; i < M * N; ++i)
            {
                err_average += errMatrix[i];
                float currErr = errMatrix[i];
                if (currErr > max_error)
                {
                    max_error = currErr;
                }
            }
            err_average /= float(M * N);

            testPassed = max_error < 0.001f;

            std::string errOutput = "";
            errOutput += "----- Matmul compare -----\n";
            errOutput += std::format("Average error (personal vs cublas): {:.6f}\n", err_average);
            errOutput += std::format("Max error(personal vs cublas): {:.6f}\n", max_error);

            if (M * N < 8192)
            {
                // personal vs cpu
                auto *h_C = (float *)malloc(M * N * sizeof(float));
                cpu_matmul(M, N, K, a, b, h_C);
                float *d_h_C;
                Wrappers::C_Malloc(&d_h_C, M * N * sizeof(float));

                Wrappers::C_Memcpy(d_h_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

                code_kernels::k_check_mat_err<<<grid_err, block_err>>>(M, N, d_C, d_h_C, d_C_err);
                Wrappers::C_GetLastError();
                Wrappers::C_DeviceSynchronize();
                float *errMatrix_cpu;
                errMatrix_cpu = new float[M * N];
                Wrappers::C_Memcpy(errMatrix_cpu, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

                float err_average_cpu = 0.0f;
                float max_error_cpu = 0.0f;
                for (int i = 0; i < M * N; ++i)
                {
                    err_average_cpu += errMatrix_cpu[i];
                    max_error_cpu = max(max_error_cpu, errMatrix_cpu[i]);
                }
                err_average_cpu /= float(M * N);
                testPassed = max_error_cpu < 0.00001;

                errOutput += std::format("Average error (personal vs cpu): {:.6f}\n", err_average_cpu);
                errOutput += std::format("Max error(personal vs cpu): {:.6f}\n", max_error_cpu);
                Wrappers::C_Free(d_h_C);
                free(h_C);
            }

            // print
            if (!testPassed)
            {
                errOutput += std::format("-FAILED-\n");
            }
            else
            {
                errOutput +=
                    std::format("Performance (personal vs cublas): {:.2f}%\n", (gflops / gflops_cublas) * 100.0);
                errOutput += std::format("-PASSED-\n");
            }

            c_matmul_benchmark_result benchmark_result;
            benchmark_result.M = M;
            benchmark_result.N = N;
            benchmark_result.K = K;
            benchmark_result.runs = runs;
            benchmark_result.personal_ms = ms_real;
            benchmark_result.personal_gflops = gflops;
            benchmark_result.cublas_ms = ms_real_cublas;
            benchmark_result.cublas_gflops = gflops_cublas;
            benchmark_result.average_error = err_average;
            benchmark_result.max_error = max_error;
            benchmark_result.passed = testPassed;

            const char *benchmark_json_path = std::getenv("CODECUDA_BENCHMARK_JSON");
            if (benchmark_json_path != nullptr && benchmark_json_path[0] != '\0')
            {
                C_SaveMatmulBenchmarkResultJson(benchmark_json_path, benchmark_result);
            }

            CODECUDA_PRINTLN(errOutput.c_str());

            Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            Wrappers::C_Free(d_A);
            Wrappers::C_Free(d_B);
            Wrappers::C_Free(d_C);
            Wrappers::C_Free(d_A_cublas);
            Wrappers::C_Free(d_B_cublas);
            Wrappers::C_Free(d_C_cublas);
            Wrappers::C_Free(d_C_err);
            Wrappers::C_EventDestroy(start);
            Wrappers::C_EventDestroy(stop);
            Wrappers::C_EventDestroy(start_cublas);
            Wrappers::C_EventDestroy(stop_cublas);
            CUBLAS_CHECK(cublasDestroy_v2(handle));
            delete[] errMatrix;
        }

    } // namespace benchmarking
    void C_Shutdown() { cublasShutdown(); }

} // namespace CodeCuda


#endif // CODECUDA_CU
