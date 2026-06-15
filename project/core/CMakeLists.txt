include(GNUInstallDirs)

add_library(codeCudaLib
        src/CodeCuda.cu
        src/CodeKernels.hpp
        src/CudaEngineInclude.hpp)

find_package(CUDAToolkit REQUIRED)

target_include_directories(codeCudaLib
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(codeCudaLib PUBLIC cxx_std_20)

set_target_properties(codeCudaLib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

target_compile_options(codeCudaLib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ptxas-options=-v>
)

target_link_libraries(codeCudaLib
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)

install(TARGETS codeCudaLib
        EXPORT CodeCudaEngineTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT CodeCudaEngineTargets
        FILE CodeCudaEngineTargets.cmake
        NAMESPACE CodeCudaEngine::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CodeCudaEngine
)
