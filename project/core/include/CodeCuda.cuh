//

// Created by carlo on 2026-01-17.
//

#ifndef CODECUDA_CUH
#define CODECUDA_CUH

namespace CodeCuda
{
    void C_Init();
    void C_Matmul(const int M, const int N, const int K, const float* a, const float* b,
                  float* c);
    void C_Matmul_Test(const int M, const int N, const int K, const float* a, const float* b,
                  float* c, int runs);
    void C_Shutdown();
    
}

#endif // CODECUDA_CUH
