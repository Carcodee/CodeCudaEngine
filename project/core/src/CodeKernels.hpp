//

// Created by carlo on 2026-06-14.
//

#ifndef CODEKERNELS_HPP
#define CODEKERNELS_HPP

namespace code_kernels
{
    namespace code_math
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
        __global__ void k_matmul_bt_row_tile(const int M, const int N, const int K, float alpha, float beta,
                                             const float *a, const float *b, float *c)
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
        __global__ void k_matmul_bt_col_tile(const int M, const int N, const int K, float alpha, float beta,
                                             const float *a, const float *b, float *c)
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
        __global__ void k_matmul_bt_2d_tilling_transposed_a(const int M, const int N, const int K, float alpha,
                                                            float beta, float *a, float *b, float *c)
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
        __global__ void k_matmul_bt_warp_tilling(const int M, const int N, const int K, float alpha, float beta,
                                                 float *a, float *b, float *c)
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
                                a_s[(dot_idx * BM) + (warp_row * WM) + (WSUBM * wm_iter_idx) +
                                    (TM * thread_row_in_warp) + tm_idx];
                        }
                    }

                    for (int wn_iter_idx = 0; wn_iter_idx < WNITER; ++wn_iter_idx)
                    {
                        for (int tn_idx = 0; tn_idx < TN; ++tn_idx)
                        {
                            reg_n[TN * wn_iter_idx + tn_idx] =
                                b_s[(dot_idx * BN) + (warp_col * WN) + (WSUBN * wn_iter_idx) +
                                    (TN * thread_col_in_warp) + tn_idx];
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
                                &c_temp[(thread_row_in_warp * TM * N) + (TN * thread_col_in_warp) + tm_idx * N])[0] =
                                tmp;
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

    } // namespace code_math

    namespace code_tests
    {

        __global__ void k_sine(int size, float time, float *data)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;
            float idxNorm = float(idx) / float(size);
            data[idx] = sin(idxNorm * time);
        }

        __global__ void k_simulation_read(int size, int sim_w, int sim_h, float min_speed, float max_speed,
                                          float avg_speed, float *u_edges, float *v_edges, float *grid_div,
                                          float *grid_pressures, float* grid_smoke,float *data)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;
            int x = idx % 1024;
            int y = idx / 1024;

            // bilinear for div and cell values
             auto uv = float2(float(x)/1023.0f, float(y)/1023.0f);
             auto pos_float = float2(uv.x * float(sim_w - 1), uv.y * float(sim_h - 1));
             
             //0 - 1 down dir in y and 0 - 1 right in x
             auto wx = pos_float.x - floor(pos_float.x);
             auto wy = pos_float.y - floor(pos_float.y);
             
             auto tl = int2(pos_float.x, pos_float.y);
             auto tr = int2(ceil(pos_float.x), pos_float.y);
             auto bl = int2(pos_float.x, ceil(pos_float.y));
             auto br = int2(ceil(pos_float.x), ceil(pos_float.y));
             
             float val_tl = grid_smoke[tl.y * sim_w + tl.x];
             float val_tr = grid_smoke[tr.y * sim_w + tr.x];
             float val_bl = grid_smoke[bl.y * sim_w + bl.x];
             float val_br = grid_smoke[br.y * sim_w + br.x];
             
             float top = (val_tl * (1.0f - wx)) + (val_tr * wx);
             float bot = (val_bl * (1.0f - wx)) + (val_br * wx);
             
             float final = (top * (1.0f - wy)) + (bot * wy);
             
             data[idx] = final;

            // bilinear for velocities
            // auto uv = float2(float(x) / 1023.0f, float(y) / 1023.0f);
            // auto pos_float = float2(uv.x * float(sim_w), uv.y * float(sim_h));
            // // 0 - 1 down dir in y and 0 - 1 right in x
            // auto wx = pos_float.x - floor(pos_float.x);
            // auto wy = pos_float.y - floor(pos_float.y);
            //
            // auto tl = int2(pos_float.x, pos_float.y);
            // auto tr = int2(ceil(pos_float.x), pos_float.y);
            // auto bl = int2(pos_float.x, ceil(pos_float.y));
            // auto br = int2(ceil(pos_float.x), ceil(pos_float.y));
            //
            // int edges_w = sim_w + 1;
            //
            // float u_tl = u_edges[tl.y * edges_w + tl.x];
            // float u_tr = u_edges[tr.y * edges_w + tr.x];
            // float u_bl = u_edges[bl.y * edges_w + bl.x];
            // float u_br = u_edges[br.y * edges_w + br.x];
            //
            // float u_top = (u_tl * (1.0f - wx)) + (u_tr * wx);
            // float u_bot = (u_bl * (1.0f - wx)) + (u_br * wx);
            // float u = (u_top * (1.0f - wy)) + (u_bot * wy);
            //
            // float v_tl = v_edges[tl.y * edges_w + tl.x];
            // float v_tr = v_edges[tr.y * edges_w + tr.x];
            // float v_bl = v_edges[bl.y * edges_w + bl.x];
            // float v_br = v_edges[br.y * edges_w + br.x];
            //
            // float v_top = (v_tl * (1.0f - wx)) + (v_tr * wx);
            // float v_bot = (v_bl * (1.0f - wx)) + (v_br * wx);
            // float v = (v_top * (1.0f - wy)) + (v_bot * wy);
            //
            // float speed = sqrtf(u * u + v * v);
            // float sign = u > 0.0 ? 1.0 : -1.0;
            //
            // float norm_speed = (speed - min_speed) / (max_speed - min_speed);
            //
            // data[idx] = speed;
        }

    } // namespace code_tests
} // namespace code_kernels

#endif // CODEKERNELS_HPP
