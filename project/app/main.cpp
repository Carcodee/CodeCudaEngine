//
// Created by carlo on 2026-01-17.
//

#include "CodeInclude.h"
#include "vector"

void TestLib(int M, int N, int K, int runs)
{
    CodeCuda::c_matrix h_a (M, N);
    CodeCuda::c_matrix h_b (N, K);
    CodeCuda::c_matrix h_c (M, K);

    h_a.Full(1.0f);
    h_b.Full(2.0f);
    h_c.Full(0.0f);

//    int matM = h_a.Shape()[0];
//    int matK = h_b.Shape()[0];
//    int matN = h_a.Shape()[1];
    
//    CodeCuda::C_Matmul(h_a.Shape()[0], h_b.Shape()[1], h_a.Shape()[1], h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data());
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
        int N = 128; 
        int K = 128; 
    };

    std::vector<MatrixSizes> sizes = {
        MatrixSizes{2, 4, 6},
        MatrixSizes{256, 28, 256},
        MatrixSizes{128, 28, 256},
        MatrixSizes{256, 28, 128},
        MatrixSizes{128, 256, 512},
        MatrixSizes{512, 256, 128},
        MatrixSizes{4098, 4098, 4098},
    };
    for (auto& size : sizes)
    {
        printf("----------------- Test with sizes: %d X %d X %d -----------------\n", size.M, size.N, size.K);
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
