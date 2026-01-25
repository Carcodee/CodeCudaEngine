//
// Created by carlo on 2026-01-17.
//

#include <iostream>
#include <string>
#include <vector>

#include "CodeCuda.cuh"
int main()
{
    CodeCuda::C_Init();
//(6 ,8)
    std::vector<float> a = {2, 2, 2, 2, 2, 2, 2, 2, 
                            2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2};

//(8 ,4)
    std::vector<float> b = {1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1,
                            1, 1, 1, 1};
    std::vector<float> c = {};
    CodeCuda::C_MatmulTest(6, 4, 8, a, b, c);

    
    for (int i = 0; i < c.size(); ++i)
    {
        if ((i) % 4 == 0)
        {
            std::cout << "\n";
        }
        std::string text = std::to_string(c[i]) + " ";
        std::cout<< text;
    }
}
