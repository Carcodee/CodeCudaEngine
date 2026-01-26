cmake_minimum_required(VERSION 3.22)

project(CodeCudaProject
        LANGUAGES CXX CUDA
)

#--Global Settings -- 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid old CUDA behavior
set(CMAKE_CUDA_ARCHITECTURES native)
# Position independent code needed for shared libs + Python modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependancies

#Subdirs
add_subdirectory(app)
add_subdirectory(core)


