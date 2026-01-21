add_executable(lib_test
        main.cpp
)

target_link_libraries(lib_test
        PRIVATE
            codeCudaLib
)

target_compile_features(lib_test PRIVATE cxx_std_17)
