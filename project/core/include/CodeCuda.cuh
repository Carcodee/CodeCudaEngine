//

// Created by carlo on 2026-01-17.
//

#ifndef CODECUDA_CUH
#define CODECUDA_CUH

#include "vector"
namespace CodeCuda
{
    void C_Init();
    void C_Matmul(const int M, const int K, const int N, const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& cOut);
    void C_Shutdown();
    void C_MatmulTest(const int M, const int K, const int N, const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& cOut);
    
}

#endif // CODECUDA_CUH
