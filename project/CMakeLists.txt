cmake_minimum_required(VERSION 3.22)

set(CODE_CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "CUDA toolkit used by CodeCudaEngine")
set(CMAKE_CUDA_COMPILER "${CODE_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler used by CodeCudaEngine" FORCE)
set(CUDAToolkit_ROOT "${CODE_CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA toolkit used by CodeCudaEngine" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "89-real" CACHE STRING "CUDA architectures used by CodeCudaEngine" FORCE)

project(CodeCudaProject
        VERSION 0.0.1
        LANGUAGES CXX CUDA
)

#--Global Settings -- 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Position independent code needed for shared libs + Python modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependancies
include(CMakePackageConfigHelpers)

#Subdirs
add_subdirectory(app)
add_subdirectory(core)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodeCudaEngineConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CodeCudaEngineConfig.cmake"
        INSTALL_DESTINATION lib/cmake/CodeCudaEngine
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CodeCudaEngineConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/CodeCudaEngineConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CodeCudaEngineConfigVersion.cmake"
        DESTINATION lib/cmake/CodeCudaEngine
)



