//
// Created by carlo on 2026-01-17.
//

#include "CodeInclude.h"
#include "vector"

void TestLib(int M, int N, int K, int runs)
{
    CodeCuda::C_Init();
    CodeCuda::c_matrix h_a (M, N);
    CodeCuda::c_matrix h_b (M, K);
    CodeCuda::c_matrix h_c (h_a.Shape()[0], h_b.Shape()[1]);

    h_a.Full(1.0f);
    h_b.Full(2.0f);
    h_c.Full(0.0f);
    
//    CodeCuda::C_Matmul(h_a.Shape()[0], h_b.Shape()[1], h_a.Shape()[1], h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data());
    CodeCuda::C_MatmulTest(h_a.Shape()[0], h_b.Shape()[1], h_a.Shape()[1], h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data(), 3);

    if (M <= 8)
    {
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
        MatrixSizes{8, 8, 8},
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
        TestLib(size.M, size.N, size.K, 5);
        printf("\n\n");
    }
    
}
int main()
{

    TestMatmulShapes();
    while (true)
    {
    }
}
