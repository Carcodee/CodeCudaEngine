//
// Created by carlo on 2026-01-17.
//

#include <iostream>
#include <vector>

#include "CodeCuda.cuh"
int main()
{
    CodeCuda::init();
    std::vector<float> a = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> b = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> c = {};
    CodeCuda::matmul(2, 3, 4, a, b, c);

    for (int i = 0; i < c.size(); ++i)
    {
        std::cout<< c[i];
    }
}
