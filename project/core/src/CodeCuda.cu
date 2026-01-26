//
// Created by carlo on 2026-01-17.
//
#ifndef CODECUDA_CU
#define CODECUDA_CU

#include <iostream>
#include "CodeInclude.h"
#include "cublas.h"
#include "cuda_runtime.h"

namespace CodeKernels
{

    __global__ void k_matmul(const int M, const int K, const int N, const float *a, const float *b, float *c)
    {
        uint32_t x_global = ((blockDim.x * blockIdx.x) + threadIdx.x);
//        printf("%d\n", x);
        auto y = static_cast<uint32_t>(x_global / K);
        if (y > M)
        {
            return;
        }
        uint32_t x = x_global % K;
        float acc = 0;
        for (int i = 0; i < N; ++i)
        {
            acc += a[N * y + i] * b[N * x + i];
//            acc += 1.0f;
        }

        c[y * K + x] = acc;
    }

    __global__ void k_matmul_naive(const int M, const int K, const int N, const float *a, const float *b, float *c)
    {
        uint32_t x_global = ((blockDim.x * blockIdx.x) + threadIdx.x);
//        printf("%d\n", x);
        auto y = static_cast<uint32_t>(x_global / K);
        if (y > M)
        {
            return;
        }
        uint32_t x = x_global % K;
        float acc = 0;
        for (int i = 0; i < N; ++i)
        {
            acc += a[N * y + i] * b[K * i + x];
//            acc += 1.0f;
        }

        c[y * K + x] = acc;
    }
} // namespace CodeKernels

namespace CodeCuda
{
    void C_Init() { std::cout << "Hello from cuda lib\n"; }
    
    void C_Matmul(const int M, const int K, const int N, const float* a, const float* b,
                  float* cOut)
    {
        if (cOut == nullptr)
        {
            std::cout << "target buffer is empty\n"; 
            return;
        }
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * N * sizeof(float));
        cudaMalloc(&d_B, N * K * sizeof(float));
        cudaMalloc(&d_C, M * K * sizeof(float));

        cudaMemcpy(d_A, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 1, 1);
        dim3 grid( int((M * K) / 32) + 1, 1, 1);
        //dim3 grid(K, M, 1);
        CodeKernels::k_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(cOut, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void C_MatmulTest(const int M, const int K, const int N, const float* a, const float* b,
                      float* cOut, int runs)
    {
        if (cOut == nullptr)
        {
            std::cout << "target buffer is empty\n"; 
            return;
        }
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * N * sizeof(float));
        cudaMalloc(&d_B, N * K * sizeof(float));
        cudaMalloc(&d_C, M * K * sizeof(float));

        cudaMemcpy(d_A, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        dim3 block(32, 1, 1);
        dim3 grid( int((M * K) / 32) + 1, 1, 1);
        // warmup
        for (int i = 0; i < 10; ++i)
        {
            CodeKernels::k_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < runs; ++i)
        {
            CodeKernels::k_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Average Time personal (ms): %f\n", ms / float(runs));

        //cubulas
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start_cublas, stop_cublas;
        cudaEventCreate(&start_cublas);
        cudaEventCreate(&stop_cublas);
        
        cublasCreate_v2(&handle);
        cudaEventRecord(start_cublas);
        for (int i = 0; i < 10; ++i)
        {
            cublasSgemm_v2(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N
            );
        }
        cudaEventRecord(stop_cublas);
        cudaEventSynchronize(stop_cublas);
        float ms_cublas = 0;
        cudaEventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
        printf("Average Time cublas (ms): %f\n", ms_cublas / 10.0f);

        cudaMemcpy(cOut, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void C_Shutdown() {}

} // namespace CodeCuda


#endif // CODECUDA_CU
