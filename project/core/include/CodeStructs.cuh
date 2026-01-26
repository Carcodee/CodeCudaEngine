//
// Created by carlo on 2026-01-22.
//

#ifndef CODESTRUCTS_CUH
#define CODESTRUCTS_CUH

namespace CodeCuda{
    
    struct c_matrix{
        c_matrix(const c_matrix&) = delete;
        c_matrix& operator=(const c_matrix&) = delete;
        c_matrix(int32_t M, int32_t N){
            this->M = M;
            this->N = N;
            this->data_size = this->M * this->N;
            this->data = new float[M * N];
            this->data_t = new float[M * N];
            this->shape = new int[2];
            this->shape[0] = M;
            this->shape[1] = N;
        }
        c_matrix& Full(float val){
            int32_t y = 0;
            for(int32_t i = 0; i < this->data_size; i++){
                data[i] = val;
            }
            return *this;
        }
        
        c_matrix& Arange_Sequential(){
            for(int32_t i = 0; i < this->data_size; i++){
                data[i] = i;
            }
            return *this;
        }
        c_matrix& BuildTranpose()
        {
            for(int32_t i = 0; i < this->data_size; i++){
                int32_t x = int32_t(i % N); 
                int32_t y = int32_t(i / N);
                data_t[x * this->M + y] = data[i];
            }
            return *this;
        }
        c_matrix& Print(){
            printf("Shape: %d x %d\n", M, N);
            std::string output= "";
            for(int32_t i = 0; i < this->data_size; i++){
                output += std::to_string(data[i]) + " ";
                if ((i + 1)% N == 0)
                {
                    output += "\n";
                }
            }
            output+= "\n";
            printf(output.c_str());
            return *this;
        }
        ~c_matrix()
        {
            delete [] shape;
            delete [] data_t;
            delete [] data;
        }
        float* Get_Data() const 
        {
            assert(data != nullptr);
            return data;
        }
        float* Get_T() 
        {
            assert(data_t != nullptr);
            BuildTranpose();
            return data_t;
        }
        
        int32_t* Shape()
        {
            return shape;
        }
    static void PrintAnyMatrix(int32_t M, int32_t N, float* data)
    {

        printf("Shape: %d x %d\n", M, N);
        std::string output = "";
        for (int32_t i = 0; i < M * N; i++)
        {
            output += std::to_string(data[i]) + " ";
            if ((i + 1) % N == 0)
            {
                output += "\n";
            }
        }
        output += "\n";
        printf(output.c_str());
    }

    private:
        float *data = nullptr;
        float *data_t = nullptr;
        int32_t M = 0;
        int32_t N = 0;
        int32_t* shape = nullptr;
        int32_t data_size = 0;
       

        
    };
    
    
}


#endif // CODESTRUCTS_CUH
