//

// Created by carlo on 2026-01-17.
//
#include <iostream>
#include "CodeCuda.cuh"
#include "cuda_runtime.h"
#include "vector"

#ifndef CODECUDA_CU
#define CODECUDA_CU

namespace Kernels
{

    __global__ void c_matmul(const int M,const int P,const int N, const float* a, const float* b, float* c)
    {
        uint32_t globalIdx = gridDim.x * blockIdx.y + blockIdx.x;
        printf("%d\n", globalIdx);
        c[globalIdx] = 1.0f;
        
    }
    
}

namespace CodeCuda
{
    void init()
    {
        std::cout<<"Hello from cuda lib\n";
    }
    void matmul(const int M, const int P, const int N, const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& cOut)
    {
        cOut.clear();
        cOut.resize(M * P);
        float *pA, *pB, *pC;
        cudaMalloc(&pA, a.size() * sizeof(float));
        cudaMalloc(&pB, b.size() * sizeof(float));
        cudaMalloc(&pC, M * P * sizeof(float));

        cudaMemcpy(pA, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(pB, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block (1, 1, 1);
        dim3 grid (M, P, 1);
        Kernels::c_matmul<<<grid, block>>>(M, P, N, pA, pB, pC);
        cudaDeviceSynchronize();

        cudaMemcpy(cOut.data(), pC, M * P * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(pA);
        cudaFree(pB);
        cudaFree(pC);
    }
    void shutdown()
    {
        
    }

}


#endif // CODECUDA_CU

