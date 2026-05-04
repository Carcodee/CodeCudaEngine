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

    // h_a.Full(1.0);
   // h_b.Full(2.0);
     h_a.Rand(1, 5);
     h_b.Rand(1, 5);
    
    h_c.Full(0.0f);
    
    
    CodeCuda::C_Matmul_Test(M, N, K, h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data(), runs);

    if (M <= 256)
    {   //h_b.Print();
        h_c.Print(true);
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
        /*
        MatrixSizes{4, 8, 4},
        MatrixSizes{256, 28, 256},
        MatrixSizes{128, 28, 256},
        MatrixSizes{256, 28, 128},
        MatrixSizes{128, 256, 512},
        MatrixSizes{512, 256, 128},
//        MatrixSizes{8098, 8098, 8098},
        */
        MatrixSizes{128, 128, 128},
        MatrixSizes{4096, 4096, 4096},
        MatrixSizes{1024, 1024, 1024},
    };
    for (auto& size : sizes)
    {
        printf("----------------- Test with sizes: %d X %d X %d -----------------\n", size.M, size.K, size.N);
        TestLib(size.M, size.N, size.K, 1);
        printf("\n\n");
    }
    
}

void Swap(std::vector<int>& list, int idx_1, int idx_2)
{
    int temp = list[idx_1];
    list[idx_1] = list[idx_2];
    list[idx_2] = temp;
}
void TwoPointerSort(std::vector<int>& list, int left_min, int left_max, int right_min, int right_max)
{
    int cur_left = left_min;
    int cur_right = right_max;
    while (cur_left < left_max || cur_right < right_max)
    {
        if (list[cur_right] < list[cur_left])
        {
            Swap(list, cur_right, cur_left);
            cur_right++;
        }else
        {
            cur_left++;
        }
    }
}

int MergeSort(std::vector<int>& list, int min, int max)
{
    if (min == max)
    {
        return min;
    }
    int new_max =min + (max - min / 2);
    int start_left = MergeSort(list, min, new_max);
    int start_right = MergeSort(list, new_max + 1, max);
    //sort
    TwoPointerSort(list, start_left, new_max, start_right, max);
    
    return min;
    
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
