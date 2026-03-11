//
// Created by carlo on 2026-02-01.
//

#ifndef CODECOMMON_HPP
#define CODECOMMON_HPP

#include "assert.h"
#include "common/Logger.hpp"

namespace CodeCommon
{
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if(err != cudaSuccess){ \
        CODECUDA_LOG_ERROR("CUDA error: ", cudaGetErrorString(err)); \
        assert(false);\
        return; \
    }\
}while(0)
    
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t err = x; \
    if(err != CUBLAS_STATUS_SUCCESS){ \
        CODECUDA_LOG_ERROR("CUBLAS error code: ", static_cast<int>(err)); \
        assert(false);\
        return; \
    }\
}while(0)
    
}
#endif // CODECOMMON_HPP
