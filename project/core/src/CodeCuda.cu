#include <chrono>
#ifndef CODECUDA_CU
#define CODECUDA_CU

#include <iostream>
#include "CodeCommon.hpp"
#include "CodeInclude.h"
#include "cublas.h"
#include "cuda_runtime.h"


namespace CodeKernels
{
    // A: M×K, B: K×N, C: M×N  (standard BLAS — K is the shared/inner dimension)
    __global__ void k_matmul(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {
        uint32_t x_global = ((blockDim.x * blockIdx.x) + threadIdx.x);
        auto y = static_cast<uint32_t>(x_global / N);
        if (y > M)
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
        if (y > M || x > N)
        {
            return;
        }
        float acc = 0;
        for (int i = 0; i < K; ++i)
        {
            acc += a[y * K + i] * b[N * i + y];
        }

        c[y * N + x] = acc;
    }
    __global__ void k_matmul_tile(const int M, const int N, const int K, float alpha, float beta, const float *a, const float *b, float *c)
    {
        uint32_t x = ((blockDim.x * blockIdx.x) + threadIdx.x);
        uint32_t y = ((blockDim.y * blockIdx.y) + threadIdx.y);
        if (y > M || x > N)
        {
            return;
        }
        float acc = 0;
        for (int i = 0; i < K; ++i)
        {
            acc += a[y * K + i] * b[N * i + x];
        }

        c[y * N + x] = acc;
    }
    __global__ void k_check_mat_err(const int M, const int N, const float *a, const float *b, float *c)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N) return;

        c[idx] = fabsf(a[idx] - b[idx]);
    }
} // namespace CodeKernels

namespace CodeCuda
{
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
        std::cout << "Hello from cuda lib\n";
    }

    void C_Matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        if (c == nullptr)
        {
            std::cout << "target buffer is empty\n";
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
        CodeKernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
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
    void C_Matmul_Test(const int M, const int N, const int K, const float *a, const float *b, float *c, int runs)
    {
        if (c == nullptr)
        {
            std::cout << "target buffer is empty\n";
            return;
        }
        

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
        dim3 block(32, 32, 1);
        dim3 grid(ceil(double(N)/32.0f), ceil(double(M)/32.0f), 1);
        
        for (int i = 0; i < 5; ++i)
        {
            CodeKernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }
        Wrappers::C_DeviceSynchronize();

        Wrappers::C_EventRecord(start);
        for (int i = 0; i < runs; ++i)
        {
            CodeKernels::k_matmul_naive<<<grid, block>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
        }
        Wrappers::C_EventRecord(stop);
        Wrappers::C_EventSynchronize(stop);
        float ms = 0;
        Wrappers::C_EventElapsedTime(&ms, start, stop);
        double ms_real = ms / double(runs);
        printf("Average Time personal (ms): %f\n", ms_real);
        auto flops = int64_t(2 * int64_t(M) * int64_t(N) * int64_t(K));
        
        double gflops = (double(flops) * 1.0e-9f) / double(ms_real / 1000.0f);
        printf("GFLOPS/s personal: %f\n", gflops);

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

        for (int i = 0; i < 5; ++i)
        {
            cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N, d_A_cublas, K, &beta, d_C_cublas, N);
        }
        Wrappers::C_EventRecord(start_cublas);
        for (int i = 0; i < runs; ++i)
        {
            cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N, d_A_cublas, K, &beta, d_C_cublas, N);
        }
        Wrappers::C_EventRecord(stop_cublas);
        Wrappers::C_EventSynchronize(stop_cublas);
        float ms_cublas = 0;
        Wrappers::C_EventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
        double ms_real_cublas = ms_cublas / double(runs);
        printf("Average Time cublas (ms): %f\n", ms_real_cublas);
        gflops = double((flops) * 1.0e-9f) / double(ms_real_cublas / 1000.0f);
        printf("GFLOPS/s cublas: %f\n", gflops);
        

        //personal vs cublas
        float *d_C_err;
        Wrappers::C_Malloc(&d_C_err, M * N * sizeof(float));

        dim3 block_err(128, 1, 1);
        dim3 grid_err(ceil(double(M * N) / 128.0), 1, 1);
        CodeKernels::k_check_mat_err<<<grid_err, block_err>>>(M, N, d_C, d_C_cublas, d_C_err);

        float *errMatrix;
        errMatrix = new float[M * N];
        Wrappers::C_Memcpy(errMatrix, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        float err_average = 0;
        float max_error = 0;
        for (int i = 0; i < M * N; ++i)
        {
            err_average+= errMatrix[i];
            max_error = max(max_error, errMatrix[i]);
        }
        err_average /= float(M * N);

        //personal vs cpu
        auto *h_C = (float*)malloc(M * N * sizeof(float));
        cpu_matmul(M, N, K, a, b, h_C);
        float* d_h_C;
        Wrappers::C_Malloc(&d_h_C, M * N * sizeof(float));

        Wrappers::C_Memcpy(d_h_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
        CodeKernels::k_check_mat_err<<<grid_err, block_err>>>(M, N, d_C, d_h_C, d_C_err);
        float *errMatrix_cpu;
        errMatrix_cpu = new float[M * N];
        Wrappers::C_Memcpy(errMatrix_cpu, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        float err_average_cpu = 0;
        float max_error_cpu = 0;
        for (int i = 0; i < M * N; ++i)
        {
            err_average_cpu += errMatrix_cpu[i];
            max_error_cpu = max(max_error_cpu, errMatrix_cpu[i]);
        }
        err_average_cpu /= float(M * N);
        
        //print
        printf("----- Matmul err compare -----\n");
        printf("Average error (personal vs cublas): %f\n", err_average);
        printf("Average error (personal vs cpu): %f\n", err_average_cpu);
        printf("Max error: %f\n", max_error);
        if (max_error > 0.00001 || max_error_cpu > 0.00001)
        {
            printf("-FAILED-\n");
        }else
        {
            printf("-PASSED-\n");
        }
        
        Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        Wrappers::C_Free(d_A);
        Wrappers::C_Free(d_B);
        Wrappers::C_Free(d_C);
        Wrappers::C_Free(d_A_cublas);
        Wrappers::C_Free(d_B_cublas);
        Wrappers::C_Free(d_C_cublas);
        Wrappers::C_Free(d_C_err);
        Wrappers::C_Free(d_h_C);
        Wrappers::C_EventDestroy(start);
        Wrappers::C_EventDestroy(stop);
        Wrappers::C_EventDestroy(start_cublas);
        Wrappers::C_EventDestroy(stop_cublas);
        CUBLAS_CHECK(cublasDestroy_v2(handle));
        delete [] errMatrix;
        free(h_C);
    }
    

    void C_Shutdown()
    {
        cublasShutdown();
    }

} // namespace CodeCuda


#endif // CODECUDA_CU
