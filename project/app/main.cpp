//
// Created by carlo on 2026-01-17.
//

#include "CodeInclude.h"
#include "vector"
void TestLib(int M, int N, int K, int runs)
{
    CodeCuda::c_matrix h_a (M, K);
    CodeCuda::c_matrix h_b (K, N);
    CodeCuda::c_matrix h_c (M, N);

    h_a.Full(1.0f);
    h_b.Full(2.0f);
    h_c.Full(0.0f);
    
    CodeCuda::C_Matmul_Test(M, N, K, h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data(), runs);

    if (M <= 8)
    {   //h_b.Print();
        h_c.Print();
    }
}
void TestMatmulShapes()
{
    struct MatrixSizes
    {
        int M = 128; 
        int K = 128; 
        int N = 128; 
    };
    

    std::vector<MatrixSizes> sizes = {
        MatrixSizes{4, 8, 4},
        MatrixSizes{256, 28, 256},
        MatrixSizes{128, 28, 256},
        MatrixSizes{256, 28, 128},
        MatrixSizes{128, 256, 512},
        MatrixSizes{512, 256, 128},
//        MatrixSizes{4098, 4098, 4098},
    };
    for (auto& size : sizes)
    {
        printf("----------------- Test with sizes: %d X %d X %d -----------------\n", size.M, size.K, size.N);
        TestLib(size.M, size.N, size.K, 1);
        printf("\n\n");
    }
    
}
int main()
{

    CodeCuda::C_Init();
    TestMatmulShapes();
    CodeCuda::C_Shutdown();
    while (true)
    {
    }
}
