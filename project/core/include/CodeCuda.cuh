//

// Created by carlo on 2026-01-17.
//

#ifndef CODECUDA_CUH
#define CODECUDA_CUH

#include "iostream"
#include "vector"
namespace CodeCuda
{
    void init();
    void matmul(const int M, const int P, const int N, const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& cOut);
    void shutdown();
}

#endif // CODECUDA_CUH
