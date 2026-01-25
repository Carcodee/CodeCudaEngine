//
// Created by carlo on 2026-01-17.
//
#ifndef CODECUDA_CU
#define CODECUDA_CU

#include <iostream>
#include "CodeCuda.cuh"
#include "cublas.h"
#include "cuda_runtime.h"
#include "vector"

namespace CodeKernels
{

    __global__ void c_matmul(const int M, const int K, const int N, const float *a, const float *b, float *c)
    {
        uint32_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
        uint32_t y = (blockDim.y * blockIdx.y) +  threadIdx.y;
        uint32_t x_global_size = gridDim.x * blockDim.x;
        if (x > K || y > M)
        {
            return;
        }
        printf("%d\n", x);
        float acc = 0;
        for (int i = 0; i < N; ++i)
        {
            acc += a[N * y + i] * b[K * i + x];
        }

        c[y * K + x] = acc;
    }

} // namespace CodeKernels

namespace CodeCuda
{
    void C_Init() { std::cout << "Hello from cuda lib\n"; }
    void C_Matmul(const int M, const int K, const int N, const std::vector<float> &a, const std::vector<float> &b,
                  std::vector<float> &cOut)
    {
        cOut.clear();
        cOut.resize(M * K);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, a.size() * sizeof(float));
        cudaMalloc(&d_B, b.size() * sizeof(float));
        cudaMalloc(&d_C, M * K * sizeof(float));

        cudaMemcpy(d_A, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(32, 1, 1);
        dim3 grid(int(K / 32) + 1, M, 1);
        CodeKernels::c_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(cOut.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void C_MatmulTest(const int M, const int K, const int N, const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &cOut)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cOut.clear();
        cOut.resize(M * K);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, a.size() * sizeof(float));
        cudaMalloc(&d_B, b.size() * sizeof(float));
        cudaMalloc(&d_C, M * K * sizeof(float));

        cudaMemcpy(d_A, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

        /*
        dim3 block(32, 1, 1);
        dim3 grid(int(K / 32) + 1, M , 1);
        // warmup
        for (int i = 0; i < 10; ++i)
        {
            CodeKernels::c_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i)
        {
            CodeKernels::c_matmul<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Average Time personal (ms): %f\n", ms / 10.0f);

        */
        //cubulas
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasHandle_t handle;
        cublasCreate_v2(&handle);
        cudaEvent_t start_cublas, stop_cublas;
        cudaEventCreate(&start_cublas);
        cudaEventCreate(&stop_cublas);
        
        cudaEventRecord(start_cublas);
        for (int i = 0; i < 10; ++i)
        {
            cublasSgemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M, N, K,
                alpha,
                d_B, K,
                d_A, N,
                beta,
                d_C, K
            );
        }
        cudaEventRecord(stop_cublas);
        cudaEventSynchronize(stop_cublas);
        float ms_cublas = 0;
        cudaEventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
        printf("Average Time cublas (ms): %f\n", ms_cublas / 10.0f);

        
        cudaMemcpy(cOut.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void C_Shutdown() {}

} // namespace CodeCuda


#endif // CODECUDA_CU
