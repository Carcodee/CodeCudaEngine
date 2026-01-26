//
// Created by carlo on 2026-01-17.
//

#include "CodeInclude.h"
int main()
{
    CodeCuda::C_Init();
    CodeCuda::c_matrix h_a (6, 4);
    CodeCuda::c_matrix h_b (4, 6);
    CodeCuda::c_matrix h_c (h_a.Shape()[0], h_b.Shape()[1]);

    h_a.Full(1.0f);
    h_b.Full(2.0f);
    h_c.Full(0.0f);
    
    CodeCuda::C_MatmulTest(h_a.Shape()[0], h_b.Shape()[1], h_a.Shape()[1], h_a.Get_Data(), h_b.Get_Data(), h_c.Get_Data(), 5);

    h_a.Print();
    h_b.Print();
    h_c.Print();

    
}
