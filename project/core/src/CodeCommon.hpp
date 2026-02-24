//
// Created by carlo on 2026-02-01.
//

#ifndef CODECOMMON_HPP
#define CODECOMMON_HPP

#include "assert.h"

namespace CodeCommon
{
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if(err != cudaSuccess){ \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        assert(false);\
        return; \
    }\
}while(0)
    
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t err = x; \
    if(err != CUBLAS_STATUS_SUCCESS){ \
        printf("CUBLAS error %s at %s:%d\n", cublasGetError(), __FILE__, __LINE__); \
        assert(false);\
        return; \
    }\
}while(0)
    
}
#endif // CODECOMMON_HPP
