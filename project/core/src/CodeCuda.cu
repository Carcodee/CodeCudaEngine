#include <functional>
#ifndef CODECUDA_CU
#define CODECUDA_CU

#include "CodeInclude.h"
#include <iostream>
#include "CodeCommon.hpp"
#include "cublas.h"
#include "cuda_runtime.h"
#include <map>
#include <chrono>


namespace code_kernels
{
    // A: M×K, B: K×N, C: M×N  (standard BLAS — K is the shared/inner dimension)
    __global__ void k_matmul(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
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

    __global__ void k_matmul_naive(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
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
    
#define BLOCKSIZE 32
    __global__ void k_matmul_x_y(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {
        //x = rows
        extern __shared__ float smem[];
        float* a_s = smem;
        float* b_s = smem + BLOCKSIZE * BLOCKSIZE;
        
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
        if (row < M && col < N) {                                                                                                                                                                               
            c[row * N + col] = tmp;                                                                                                                                                                             
        }
    }
    
    
    
    __global__ void k_matmul_bt_row_tile(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {

        constexpr int BM = 64;
        constexpr int BK = 8;
        constexpr int BN = 64;
        //x = rows
        extern __shared__ float smem[];
        
        float* a_s = smem;
        float* b_s = smem + (BM * BK);
        
        uint32_t row = blockIdx.x * BM + (threadIdx.x / (BK));
        uint32_t local_col = (threadIdx.x % (BK));

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);
        
        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);
        int b_stride = BN/BK;
        
        a += blockIdx.x * BM * K;
        b += blockIdx.y * b_stride;

        float bt[BK] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        //k
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
        
        if (row < M) {
        }
            int total_c_entries_per_block = (blockIdx.y * BK * BK);
            //each threat cal 0 ... 8 ... 16 ... 24 ... 32 ... etc
            for (int col_offset = 0; col_offset < BK; ++col_offset)
            {
                c[(row * N) + (BK * col_offset) + total_c_entries_per_block + local_col] = bt[col_offset]; 
            }
        
    }
    
    __global__ void k_matmul_bt_col_tile(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {
        
        constexpr int BM = 64;
        constexpr int BK = 8;
        constexpr int BN = 64;
        //x = rows
        //y = rows
        extern __shared__ float smem[];
        
        float* a_s = smem;
        float* b_s = smem + (BM * BK);
        
        uint32_t row_thread = threadIdx.x / BN;
        uint32_t col_thread = threadIdx.x % BN;

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);
        
        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);
        
        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        c += (blockIdx.y * BM * K) + (blockIdx.x * BN);

        float bt[BK] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        //k
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
    __global__ void k_matmul_bt_2d_tilling(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {
        //x = rows
        //y = rows
        constexpr uint32_t BM = 64;
        constexpr uint32_t BK = 8;
        constexpr uint32_t BN = 64;
        constexpr uint32_t TM =8;
        constexpr uint32_t TN =8;
        
        extern __shared__ float smem[];
        
        float* a_s = smem;
        float* b_s = smem + (BM * BK);
        
        uint32_t row_thread = threadIdx.x / BK;
        uint32_t col_thread = threadIdx.x % BK;

        uint32_t local_row_a = (threadIdx.x / BK);
        uint32_t local_col_a = (threadIdx.x % BK);
        
        uint32_t local_row_b = (threadIdx.x / BN);
        uint32_t local_col_b = (threadIdx.x % BN);
        
        a += blockIdx.y * BM * K;
        b += blockIdx.x * BN;
        c += (blockIdx.y * BM * K) + (blockIdx.x * BN);

        float bt[TM * TN] = {0.0};
        float reg_m[TM] = {0.0};
        float reg_n[TN] = {0.0};

        int tile_count = ceilf(float(K) / float(BK));

        //keep in mind for this matmul algorithms there is a lot of sizes that need to match
        // in this case is not casuality that BK * BN/BK * BK = 8, that means we can safely jump on 
        //a and b by BM * K and in B by BN
        //this is not fully loading all the smem
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
                for (int reg_n_idx= 0; reg_n_idx < TN; reg_n_idx++)
                {
                    for (int reg_m_idx = 0; reg_m_idx < TM; ++reg_m_idx)
                    {
                        //todo
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
                c[(row_thread * BK + row_offset) * N + (col_thread * BK) + col_offset] = bt[row_offset * BK + col_offset];
            }
        }
    }
    __global__ void k_check_mat_err(const int M, const int N, const float *a, const float *b, float *c)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N) return;
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
        
    }
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

        dim3 block(32, 1, 1);
        dim3 grid(int((M * N) / 32) + 1, 1, 1);
        code_kernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        Wrappers::C_GetLastError();
        Wrappers::C_DeviceSynchronize();

        Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        Wrappers::C_Free(d_A);
        Wrappers::C_Free(d_B);
        Wrappers::C_Free(d_C);
    }

    void cpu_matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        for (int i = 0; i <M; ++i)
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

    void add_kernel_launcher(const std::string& name, std::function<void()> kernelFunc, std::map<std::string, Internals::kernel_launcher>& kernels_out)
    {
        Internals::kernel_launcher launcher;
        launcher.kernel = std::move(kernelFunc);
        kernels_out.try_emplace(name, launcher);
    }
    
    void C_Matmul_Test(const int M, const int N, const int K, const float *a, const float *b, float *c, int runs)
    {
        if (c == nullptr)
        {
            CODECUDA_LOG_WARNING("target buffer is empty");
            return;
        }
        bool testPassed = true;

        //personal
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
        
        add_kernel_launcher("naive_coalescent", [N, M, K, d_A, d_B, d_C]()
        {
            dim3 grid(ceil(double(N)/32.0f), ceil(double(M)/32.0f));
            dim3 block(32, 32);
            code_kernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }, kernels);
        add_kernel_launcher("smem", [N, M, K, d_A, d_B, d_C]()
        {
            dim3 grid(ceil(double(M)/ 32.0f), ceil(double(N)/32.0f));
            dim3 block(32 * 32);
            code_kernels::k_matmul_x_y<<<grid, block, block.x * sizeof(float) * 2>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }, kernels);
        
        add_kernel_launcher("b_tilling_col_tile", [N, M, K, d_A, d_B, d_C]()
        {
            constexpr uint32_t BM = 64;
            constexpr uint32_t BK = 8;
            constexpr uint32_t BN = 64;
            dim3 grid(ceil(double(N)/double(BN)), ceil(double(M)/double(BK * BK)));
            dim3 block(BM * BK);
            code_kernels::k_matmul_bt_col_tile<<<grid, block, (BN * BK + BK * BM) * sizeof(float)>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }, kernels);
        
        add_kernel_launcher("b_tilling_row_tile", [N, M, K, d_A, d_B, d_C]()
        {
            
            constexpr uint32_t BM = 64;
            constexpr uint32_t BK = 8;
            constexpr uint32_t BN = 64;
            dim3 grid(ceil(double(M)/double(BM)), ceil(double(N)/double(BK * BK)));
            dim3 block(BM * BK);
            code_kernels::k_matmul_bt_row_tile<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }, kernels);
        
        add_kernel_launcher("b_tilling_2d", [N, M, K, d_A, d_B, d_C]()
        {
            constexpr uint32_t BM = 64;
            constexpr uint32_t BK = 8;
            constexpr uint32_t BN = 64;
            dim3 grid(ceil(double(M)/double(BK * BK)), ceil(double(N)/double(BK * BK)));
            dim3 block(BK * BK);
            code_kernels::k_matmul_bt_2d_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float)>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }, kernels);
        


        for (int i = 0; i < 5; ++i)
        {
            kernels.at("b_tilling_2d").kernel();
        }
        Wrappers::C_GetLastError();
        Wrappers::C_DeviceSynchronize();

        Wrappers::C_EventRecord(start);
        for (int i = 0; i < runs; ++i)
        {
            kernels.at("b_tilling_2d").kernel();
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

        //
        // for (int i = 0; i < runs; ++i)
        // {
        //     kernels.at("b_tilling_row_tile").kernel();
        // }
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
        
        add_kernel_launcher("cublas", [handle, N, M, K, alpha,d_B_cublas, d_A_cublas, beta, d_C_cublas]()
        {
            CUBLAS_CHECK(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N, d_A_cublas, K, &beta, d_C_cublas, N));
        }, kernels);

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
        gflops = double((flops) * 1.0e-9f) / double(ms_real_cublas / 1000.0f);
        CODECUDA_PRINTLN("GFLOPS/s cublas: ", gflops);
        

        //personal vs cublas
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

        testPassed = max_error < 0.00001f;
        
        std::string errOutput = "";
        errOutput+="----- Matmul err compare -----\n";
        errOutput+= std::format("Average error (personal vs cublas): {:.6f}\n", err_average);
        errOutput+= std::format("Max error(personal vs cublas): {:.6f}\n", max_error);
        
        if (M * N < 10000)
        {
            //personal vs cpu
            auto *h_C = (float*)malloc(M * N * sizeof(float));
            cpu_matmul(M, N, K, a, b, h_C);
            float* d_h_C;
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
                
            errOutput+= std::format("Average error (personal vs cpu): {:.6f}\n", err_average_cpu);
            errOutput+= std::format("Max error(personal vs cpu): {:.6f}\n", max_error_cpu);
            Wrappers::C_Free(d_h_C);
            free(h_C);
        }
        
        //print
        if (!testPassed)
        {
            errOutput+= std::format("-FAILED-\n");
        }else
        {
            errOutput+= std::format("-PASSED-\n");
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
        delete [] errMatrix;
    }
    

    void C_Shutdown()
    {
        cublasShutdown();
    }

} // namespace CodeCuda


#endif // CODECUDA_CU
