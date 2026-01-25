add_library(codeCudaLib src/CodeCuda.cu)

find_package(CUDAToolkit REQUIRED)

target_include_directories(codeCudaLib 
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(codeCudaLib PUBLIC cxx_std_20)

set_target_properties(codeCudaLib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

target_compile_options(codeCudaLib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(codeCudaLib
        PRIVATE
        CUDA::cudart
        CUDA::cublas
)


