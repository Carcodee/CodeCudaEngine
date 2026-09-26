cmake_minimum_required(VERSION 3.22)

project(CodeCudaProject
        VERSION 0.0.1
        LANGUAGES CXX CUDA
)

# Match consumers that link the distributed library with binary dependencies.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

#--Global Settings -- 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "all-major" CACHE STRING
    "CUDA architectures used by CodeCudaEngine (all compiler-supported major architectures plus forward-compatible PTX)")

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



