//

// Created by carlo on 2026-01-17.
//

#ifndef CODECUDA_CUH
#define CODECUDA_CUH

#include <cstdint>

namespace CodeCuda
{
    void C_Init();
    void C_Matmul(const int M, const int N, const int K, const float* a, const float* b,
                  float* c);

    void C_Shutdown();
    
    namespace CodeBenchmarking
    {
        
        struct c_matmul_benchmark_result
        {
            int32_t M = 0;
            int32_t N = 0;
            int32_t K = 0;
            int32_t runs = 0;
            double personal_ms = 0.0;
            double personal_gflops = 0.0;
            double cublas_ms = 0.0;
            double cublas_gflops = 0.0;
            double average_error = 0.0;
            double max_error = 0.0;
            bool passed = false;
        };

        void C_Matmul_Test(const int M, const int N, const int K, const float* a, const float* b,
                      float* c, int runs);
        void C_SaveMatmulBenchmarkResultJson(const char* path, const c_matmul_benchmark_result& result);
    }
    
}

#endif // CODECUDA_CUH
