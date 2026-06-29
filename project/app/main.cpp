//
// Created by carlo on 2026-01-17.
//

#include "../core/src/common/Logger.hpp"
#include "CodeInclude.h"
#include "vector"
void TestLib(CodeCuda::CodeCudaContext* context,int M, int N, int K, int runs)
{
    CodeCuda::c_matrix h_a (M, K);
    CodeCuda::c_matrix h_b (K, N);
    CodeCuda::c_matrix h_c (M, N);

    // h_a.Full(1.0);
   // h_b.Full(2.0);
     h_a.RandInt(10, 50);
     h_b.RandInt(10, 50);
    
    h_c.Full(0.0f);
    
    CODECUDA_PRINTLN("s", h_c.Get_Data()[0]);
    
    CodeCuda::CodeBenchmarking::C_Matmul_Test(context, M, N, K, h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data(), runs);

    if (M <= 256)
    {   
        h_c.Print(false, true);
    }
}
void TestMatmulShapes(CodeCuda::CodeCudaContext* context)
{
    struct MatrixSizes
    {
        int M = 128; 
        int K = 128; 
        int N = 128; 
    };
    

    std::vector<MatrixSizes> sizes = {
//         /*
//         MatrixSizes{4, 8, 4},
//         MatrixSizes{256, 28, 256},
//         MatrixSizes{128, 28, 256},
//         MatrixSizes{256, 28, 128},
//         MatrixSizes{128, 256, 512},
//         MatrixSizes{512, 256, 128},
//         */
        
        MatrixSizes{128, 128, 128},
        MatrixSizes{256, 256, 256},
        MatrixSizes{512, 512, 512},
        MatrixSizes{1024, 1024, 1024},
        MatrixSizes{2048, 2048, 2048},
        MatrixSizes{4096, 4096, 4096},
        // MatrixSizes{8192, 8192, 8192},
    };
    for (auto& size : sizes)
    {
        printf("----------------- Test with sizes: %d X %d X %d -----------------\n", size.M, size.K, size.N);
        TestLib(context, size.M, size.N, size.K, 15);
        printf("\n\n");
    }
    
}
int main()
{
    auto cuda_context = new CodeCuda::CodeCudaContext();
    CodeCuda::C_Init(cuda_context);
    TestMatmulShapes(cuda_context);
    CodeCuda::C_Shutdown(cuda_context);
    while (true)
    {
    }
}
