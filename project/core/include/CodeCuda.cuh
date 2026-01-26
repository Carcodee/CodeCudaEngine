//

// Created by carlo on 2026-01-17.
//

#ifndef CODECUDA_CUH
#define CODECUDA_CUH

namespace CodeCuda
{
    void C_Init();
    void C_Matmul(const int M, const int K, const int N, const float* a, const float* b,
                  float* cOut);
    void C_MatmulTest(const int M, const int K, const int N, const float* a, const float* b,
                  float* cOut, int runs);
    void C_Shutdown();
    
}

#endif // CODECUDA_CUH
