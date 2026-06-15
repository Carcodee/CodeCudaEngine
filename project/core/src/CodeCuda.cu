
#ifndef CODECUDA_CU
#define CODECUDA_CU


#include "CudaEngineInclude.hpp"


namespace Wrappers
{
    inline void C_Free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }
    inline void C_DeviceSynchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }
    inline void C_GetLastError() { CUDA_CHECK(cudaGetLastError()); }
    inline void C_FreeZero() { CUDA_CHECK(cudaFree(0)); }
    inline void C_StreamCreate(cudaStream_t *stream) { CUDA_CHECK(cudaStreamCreate(stream)); }
    inline void C_StreamDestroy(cudaStream_t stream) { CUDA_CHECK(cudaStreamDestroy(stream)); }
    inline void C_StreamSynchronize(cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    inline void C_EventCreate(cudaEvent_t *event) { CUDA_CHECK(cudaEventCreate(event)); }
    inline void C_EventDestroy(cudaEvent_t event) { CUDA_CHECK(cudaEventDestroy(event)); }
    inline void C_EventRecord(cudaEvent_t event, cudaStream_t stream = 0)
    {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
    inline void C_EventSynchronize(cudaEvent_t event) { CUDA_CHECK(cudaEventSynchronize(event)); }
    inline void C_EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)
    {
        CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
    }
    inline void C_Memset(void *dst, int value, size_t count) { CUDA_CHECK(cudaMemset(dst, value, count)); }
    inline void C_SetDevice(int device) { CUDA_CHECK(cudaSetDevice(device)); }
    inline void C_GetDevice(int *device) { CUDA_CHECK(cudaGetDevice(device)); }
    inline void C_GetDeviceCount(int *device) { CUDA_CHECK(cudaGetDeviceCount(device)); }
    inline void C_GetDeviceProperties(cudaDeviceProp *prop, int device)
    {
        CUDA_CHECK(cudaGetDeviceProperties(prop, device));
    }
    template <class T>
    inline void C_HostMalloc(T **ptr, size_t size, unsigned int flags = cudaHostAllocDefault)
    {
        CUDA_CHECK(cudaHostAlloc((void **)ptr, size, flags));
    }
    inline void C_HostFree(void *ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
    template <class T>
    inline void C_MallocManaged(T **ptr, size_t size, unsigned int flags = cudaMemAttachGlobal)
    {
        CUDA_CHECK(cudaMallocManaged((void **)ptr, size, flags));
    }
    template <class T>
    inline void C_Malloc(T **ptr, size_t size)
    {
        CUDA_CHECK(cudaMalloc((void **)ptr, size));
    }
    inline void C_Memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
    }
    inline void C_MemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, count, kind, stream));
    }
    inline void C_DeviceReset() { CUDA_CHECK(cudaDeviceReset()); }

} // namespace Wrappers

namespace CodeCuda
{
    class CudaContext
    {
    public:
        cudaStream_t stream = nullptr;
        int device = -1;
        bool initialized = false;
    };

    inline CudaContext cuda_context{};

    C_Res C_Init()
    {
        CODECUDA_PRINTLN("Initialized codeCudaLib");
        Wrappers::C_FreeZero();

        int device_count = 0;
        Wrappers::C_GetDeviceCount(&device_count);

        assert(device_count > 0 && "There is no valid device for CodeCudaEngine");
        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp props{};

            Wrappers::C_GetDeviceProperties(&props, i);
            if (i == 0)
            {
                CODECUDA_PRINTLN("Selected-> ", props.name);
            }
            CODECUDA_PRINTLN(props.name);
            Wrappers::C_GetDeviceProperties(&props, 0);

            CODECUDA_PRINTLN("CUDA Device: ", props.name);
            CODECUDA_PRINTLN("SM Count: ", props.multiProcessorCount);
            CODECUDA_PRINTLN("Shared Mem Per Block: ", props.sharedMemPerBlock);
        }


        Wrappers::C_SetDevice(0);

        Wrappers::C_StreamCreate(&cuda_context.stream);
        cuda_context.initialized = true;
        return C_Res::OK;
    }
    C_Res C_InitFromExternalDevice(uint8_t *vkDeviceUUID, size_t UUID_SIZE)
    {
        Wrappers::C_FreeZero();

        int current_device = 0;
        int device_count = 0;
        int devices_prohibited = 0;

        Wrappers::C_GetDeviceCount(&device_count);
        assert(device_count > 0 && "There is no valid device for CodeCudaEngine");

        cudaDeviceProp device_prop{};
        while (current_device < device_count)
        {
            Wrappers::C_GetDeviceProperties(&device_prop, current_device);

            if (device_prop.computeMode != cudaComputeModeProhibited)
            {
                int ret = memcmp((void *)&device_prop.uuid, vkDeviceUUID, UUID_SIZE);
                if (ret == 0)
                {
                    Wrappers::C_GetDeviceProperties(&device_prop, current_device);
                    break;
                }
            }
            else
            {
                devices_prohibited++;
            }
            current_device++;
        }
        if (devices_prohibited == device_count) {
            CODECUDA_PRINTLN("CUDA error: No Vulkan-CUDA Interop capable GPU found.");
            return C_Res::ERR;
        }
        
        Wrappers::C_SetDevice(current_device);
        Wrappers::C_StreamCreate(&cuda_context.stream);
        cuda_context.initialized = true;
        CODECUDA_PRINTLN("Initialized -------CodeCudaEngine-------");
        CODECUDA_PRINTLN("GPU Device index: ", current_device, "\nName: ", device_prop.name, " \nWith compute capability: ", device_prop.major, device_prop.minor);
        return C_Res::OK;
        
    }

    C_Res C_Shutdown()
    {
        CODECUDA_PRINTLN("Shutdown codeCudaLib");
        if (!cuda_context.initialized)
            return C_Res::ERR;

        Wrappers::C_StreamSynchronize(cuda_context.stream);
        Wrappers::C_StreamDestroy(cuda_context.stream);

        cuda_context.stream = nullptr;
        cuda_context.device = -1;
        cuda_context.initialized = false;
        return C_Res::OK;
    }

    C_Res C_Matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        if (c == nullptr)
        {
            CODECUDA_LOG_WARNING("target buffer is empty");
            return C_Res::ERR;
        }
        float *d_A, *d_B, *d_C;
        Wrappers::C_Malloc(&d_A, M * K * sizeof(float));
        Wrappers::C_Malloc(&d_B, K * N * sizeof(float));
        Wrappers::C_Malloc(&d_C, M * N * sizeof(float));

        Wrappers::C_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
        Wrappers::C_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

        constexpr uint32_t BM = k_auto_tunning_params::BM;
        constexpr uint32_t BK = k_auto_tunning_params::BK;
        constexpr uint32_t BN = k_auto_tunning_params::BN;
        constexpr uint32_t BSIZE = k_auto_tunning_params::BSIZE;

        dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
        dim3 block(BSIZE);
        code_kernels::code_math::
            k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float), cuda_context.stream>>>(
                M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);

        Wrappers::C_GetLastError();
        Wrappers::C_DeviceSynchronize();

        Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        Wrappers::C_Free(d_A);
        Wrappers::C_Free(d_B);
        Wrappers::C_Free(d_C);
        
        return C_Res::OK;
    }

    void cpu_matmul(const int M, const int N, const int K, const float *a, const float *b, float *c)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float temp = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    temp += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = temp;
            }
        }
    }

    namespace CodeBenchmarking
    {
        void C_Matmul_Test(const int M, const int N, const int K, const float *a, const float *b, float *c, int runs)
        {
            if (c == nullptr)
            {
                CODECUDA_LOG_WARNING("target buffer is empty");
                return;
            }

            cublasInit();
            bool testPassed = true;

            // personal
            float *d_A, *d_B, *d_C;
            Wrappers::C_Malloc(&d_A, M * K * sizeof(float));
            Wrappers::C_Malloc(&d_B, K * N * sizeof(float));
            Wrappers::C_Malloc(&d_C, M * N * sizeof(float));

            Wrappers::C_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            Wrappers::C_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            Wrappers::C_EventCreate(&start);
            Wrappers::C_EventCreate(&stop);
            /*
            dim3 grid(ceil(double(N)/32.0f), ceil(double(M)/32.0f));
            dim3 block(32, 32);
            */

            std::map<std::string, Internals::kernel_launcher> kernels;

            using namespace code_kernels::code_math;
            Internals::add_kernel_launcher(
                "naive_coalescent",
                [N, M, K, d_A, d_B, d_C]()
                {
                    dim3 grid(ceil(double(N) / 32.0f), ceil(double(M) / 32.0f));
                    dim3 block(32, 32);
                    k_matmul_naive<<<grid, block, 0, cuda_context.stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);
            Internals::add_kernel_launcher(
                "smem",
                [N, M, K, d_A, d_B, d_C]()
                {
                    dim3 grid(ceil(double(M) / 32.0f), ceil(double(N) / 32.0f));
                    dim3 block(32 * 32);
                    k_matmul_x_y<<<grid, block, block.x * sizeof(float) * 2, cuda_context.stream>>>(M, N, K, 1.0f, 0.0f,
                                                                                                    d_A, d_B, d_C);
                },
                kernels);

            Internals::add_kernel_launcher(
                "b_tilling_col_tile",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BM * BK);
                    k_matmul_bt_col_tile<<<grid, block, (BN * BK + BK * BM) * sizeof(float), cuda_context.stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            Internals::add_kernel_launcher(
                "b_tilling_row_tile",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(M) / double(BM)), ceil(double(N) / double(BK * BK)));
                    dim3 block(BM * BK);
                    k_matmul_bt_row_tile<<<grid, block, (BM * BK + BK * BN) * sizeof(float), cuda_context.stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            Internals::add_kernel_launcher(
                "b_tilling_2d",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    k_matmul_bt_2d_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float), cuda_context.stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            Internals::add_kernel_launcher(
                "b_tilling_2d_transposed",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    k_matmul_bt_2d_tilling_transposed_a<<<grid, block, (BM * BK + BK * BN) * sizeof(float),
                                                          cuda_context.stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            Internals::add_kernel_launcher(
                "warp_tilling",
                [N, M, K, d_A, d_B, d_C]()
                {
                    constexpr uint32_t BM = k_auto_tunning_params::BM;
                    constexpr uint32_t BK = k_auto_tunning_params::BK;
                    constexpr uint32_t BN = k_auto_tunning_params::BN;
                    constexpr uint32_t BSIZE = k_auto_tunning_params::BSIZE;

                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
                    dim3 block(BSIZE);
                    k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float), cuda_context.stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);


            for (int i = 0; i < 5; ++i)
            {
                kernels.at("warp_tilling").kernel();
            }
            Wrappers::C_GetLastError();
            Wrappers::C_DeviceSynchronize();

            Wrappers::C_EventRecord(start);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("warp_tilling").kernel();
            }
            Wrappers::C_GetLastError();

            Wrappers::C_EventRecord(stop);
            Wrappers::C_EventSynchronize(stop);
            float ms = 0;
            Wrappers::C_EventElapsedTime(&ms, start, stop);
            double ms_real = ms / double(runs);
            CODECUDA_PRINTLN("Average Time personal (ms): ", ms_real);
            auto flops = int64_t(2 * int64_t(M) * int64_t(N) * int64_t(K));

            double gflops = (double(flops) * 1.0e-9f) / double(ms_real / 1000.0f);
            CODECUDA_PRINTLN("GFLOPS/s personal: ", gflops);

            // cubulas
            float *d_A_cublas, *d_B_cublas, *d_C_cublas;
            Wrappers::C_Malloc(&d_A_cublas, M * K * sizeof(float));
            Wrappers::C_Malloc(&d_B_cublas, K * N * sizeof(float));
            Wrappers::C_Malloc(&d_C_cublas, M * N * sizeof(float));

            Wrappers::C_Memcpy(d_A_cublas, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            Wrappers::C_Memcpy(d_B_cublas, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            float alpha = 1.0f;
            float beta = 0.0f;
            cublasHandle_t handle;
            cudaEvent_t start_cublas, stop_cublas;
            Wrappers::C_EventCreate(&start_cublas);
            Wrappers::C_EventCreate(&stop_cublas);

            CUBLAS_CHECK(cublasCreate_v2(&handle));

            Internals::add_kernel_launcher(
                "cublas",
                [handle, N, M, K, alpha, d_B_cublas, d_A_cublas, beta, d_C_cublas]()
                {
                    CUBLAS_CHECK(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N,
                                                d_A_cublas, K, &beta, d_C_cublas, N));
                },
                kernels);

            for (int i = 0; i < 5; ++i)
            {
                kernels.at("cublas").kernel();
            }
            Wrappers::C_EventRecord(start_cublas);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("cublas").kernel();
            }
            Wrappers::C_EventRecord(stop_cublas);
            Wrappers::C_EventSynchronize(stop_cublas);
            float ms_cublas = 0;
            Wrappers::C_EventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
            double ms_real_cublas = ms_cublas / double(runs);
            CODECUDA_PRINTLN("Average Time cublas (ms): ", ms_real_cublas);
            double gflops_cublas = double(double(flops) * 1.0e-9) / double(ms_real_cublas / 1000.0);
            CODECUDA_PRINTLN("GFLOPS/s cublas: ", gflops_cublas);
            // personal vs cublas
            float *d_C_err;
            Wrappers::C_Malloc(&d_C_err, M * N * sizeof(float));

            dim3 block_err(128, 1, 1);
            dim3 grid_err(ceil(double(M * N) / 128.0), 1, 1);
            k_check_mat_err<<<grid_err, block_err, 0, cuda_context.stream>>>(M, N, d_C, d_C_cublas, d_C_err);
            Wrappers::C_GetLastError();
            Wrappers::C_DeviceSynchronize();

            float *errMatrix;
            errMatrix = new float[M * N];
            Wrappers::C_Memcpy(errMatrix, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            float err_average = 0.0f;
            float max_error = 0.0f;
            for (int i = 0; i < M * N; ++i)
            {
                err_average += errMatrix[i];
                float currErr = errMatrix[i];
                if (currErr > max_error)
                {
                    max_error = currErr;
                }
            }
            err_average /= float(M * N);

            testPassed = max_error < 0.001f;

            std::string errOutput = "";
            errOutput += "----- Matmul compare -----\n";
            errOutput += std::format("Average error (personal vs cublas): {:.6f}\n", err_average);
            errOutput += std::format("Max error(personal vs cublas): {:.6f}\n", max_error);

            if (M * N < 8192)
            {
                // personal vs cpu
                auto *h_C = (float *)malloc(M * N * sizeof(float));
                cpu_matmul(M, N, K, a, b, h_C);
                float *d_h_C;
                Wrappers::C_Malloc(&d_h_C, M * N * sizeof(float));

                Wrappers::C_Memcpy(d_h_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

                k_check_mat_err<<<grid_err, block_err, 0, cuda_context.stream>>>(M, N, d_C, d_h_C, d_C_err);
                Wrappers::C_GetLastError();
                Wrappers::C_DeviceSynchronize();
                float *errMatrix_cpu;
                errMatrix_cpu = new float[M * N];
                Wrappers::C_Memcpy(errMatrix_cpu, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

                float err_average_cpu = 0.0f;
                float max_error_cpu = 0.0f;
                for (int i = 0; i < M * N; ++i)
                {
                    err_average_cpu += errMatrix_cpu[i];
                    max_error_cpu = max(max_error_cpu, errMatrix_cpu[i]);
                }
                err_average_cpu /= float(M * N);
                testPassed = max_error_cpu < 0.00001;

                errOutput += std::format("Average error (personal vs cpu): {:.6f}\n", err_average_cpu);
                errOutput += std::format("Max error(personal vs cpu): {:.6f}\n", max_error_cpu);
                Wrappers::C_Free(d_h_C);
                free(h_C);
            }

            // print
            if (!testPassed)
            {
                errOutput += std::format("-FAILED-\n");
            }
            else
            {
                errOutput +=
                    std::format("Performance (personal vs cublas): {:.2f}%\n", (gflops / gflops_cublas) * 100.0);
                errOutput += std::format("-PASSED-\n");
            }

            ::CodeBenchmarking::c_matmul_benchmark_result benchmark_result;
            benchmark_result.M = M;
            benchmark_result.N = N;
            benchmark_result.K = K;
            benchmark_result.runs = runs;
            benchmark_result.personal_ms = ms_real;
            benchmark_result.personal_gflops = gflops;
            benchmark_result.cublas_ms = ms_real_cublas;
            benchmark_result.cublas_gflops = gflops_cublas;
            benchmark_result.average_error = err_average;
            benchmark_result.max_error = max_error;
            benchmark_result.passed = testPassed;

            const char *benchmark_json_path = std::getenv("CODECUDA_BENCHMARK_JSON");
            if (benchmark_json_path != nullptr && benchmark_json_path[0] != '\0')
            {
                C_SaveMatmulBenchmarkResultJson(benchmark_json_path, benchmark_result);
            }

            CODECUDA_PRINTLN(errOutput.c_str());

            Wrappers::C_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            Wrappers::C_Free(d_A);
            Wrappers::C_Free(d_B);
            Wrappers::C_Free(d_C);
            Wrappers::C_Free(d_A_cublas);
            Wrappers::C_Free(d_B_cublas);
            Wrappers::C_Free(d_C_cublas);
            Wrappers::C_Free(d_C_err);
            Wrappers::C_EventDestroy(start);
            Wrappers::C_EventDestroy(stop);
            Wrappers::C_EventDestroy(start_cublas);
            Wrappers::C_EventDestroy(stop_cublas);
            CUBLAS_CHECK(cublasDestroy_v2(handle));
            delete[] errMatrix;
            cublasShutdown();
        }
    } // namespace CodeBenchmarking


} // namespace CodeCuda


#endif // CODECUDA_CU
