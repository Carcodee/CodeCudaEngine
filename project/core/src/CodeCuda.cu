
#ifndef CODECUDA_CU
#define CODECUDA_CU


#include "CudaEngineInclude.hpp"


namespace CodeCuda
{


    static CodeSimulation::c_grid simulation = {};
    static void *grid_pressures_d;
    static void *grid_div_d;
    static void *grid_v_d;
    static void *u_d;
    static void *v_d;
    void add_kernel_launcher(const std::string &name, std::function<void(cudaStream_t)> kernelFunc,
                             std::map<std::string, kernel_launcher> &kernels_out)
    {
        kernel_launcher launcher;
        launcher.kernel = std::move(kernelFunc);
        kernels_out.try_emplace(name, launcher);
    }


    C_Res CodeCudaContext::C_Init()
    {
        CODECUDA_PRINTLN("Starting CodeCudaEngine Init");
        CODE_API::CW_FreeZero();

        int device_count = 0;
        CODE_API::CW_GetDeviceCount(&device_count);

        assert(device_count > 0 && "There is no valid device for CodeCudaEngine");
        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp props{};

            CODE_API::CW_GetDeviceProperties(&props, i);
            if (i == 0)
            {
                CODECUDA_PRINTLN("CUDA Device: ", props.name, " <- Selected");
            }
            else
            {
                CODECUDA_PRINTLN("CUDA Device: ", props.name);
            }
            CODE_API::CW_GetDeviceProperties(&props, 0);

            CODECUDA_PRINTLN("SM Count: ", props.multiProcessorCount);
            CODECUDA_PRINTLN("Shared Mem Per Block: ", props.sharedMemPerBlock);
        }


        CODE_API::CW_SetDevice(0);

        CODE_API::CW_StreamCreate(&this->stream);
        this->initialized = true;
        simulation.InitGrid(50, 50);
        CODECUDA_PRINTLN("Initialized: CodeCudaEngine");
        CODECUDA_PRINTLN("");
        CODECUDA_PRINTLN("");
        return C_Res::OK;
    }

    C_Res CodeCudaContext::C_WaitExternalSemaphore(uint64_t wait_value)
    {
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.params.fence.value = wait_value;
        CODE_API::CW_WaitExternalSemaphoresAsync(&this->external_semaphore, &waitParams, 1, this->stream);
        return C_Res::OK;
    }

    C_Res CodeCudaContext::C_SignalExternalSemaphore(uint64_t signal_value)
    {
        cudaExternalSemaphoreSignalParams signal_params{};
        signal_params.params.fence.value = signal_value;
        CODE_API::CW_SignalExternalSemaphoresAsync(&this->external_semaphore, &signal_params, 1, this->stream);
        return C_Res::OK;
    }
    C_Res CodeCudaContext::C_ExecuteKernel()
    {
        this->kernel_launcher.kernel(this->stream);
        return C_Res::OK;
    }
    C_Res CodeCudaContext::C_ExecuteCPU()
    {
        this->cpu_launcher.task();
        return C_Res::OK;
    }
    C_Res CodeCudaContext::C_InitFromExternalDevice(uint8_t *vkDeviceUUID, size_t UUID_SIZE)
    {
        CODECUDA_PRINTLN("Starting CodeCudaEngine Init");
        CODE_API::CW_FreeZero();

        int current_device = 0;
        int device_count = 0;
        int devices_prohibited = 0;

        CODE_API::CW_GetDeviceCount(&device_count);
        assert(device_count > 0 && "There is no valid device for CodeCudaEngine");

        cudaDeviceProp device_prop{};
        while (current_device < device_count)
        {
            CODE_API::CW_GetDeviceProperties(&device_prop, current_device);

            if (device_prop.computeMode != cudaComputeModeProhibited)
            {
                int ret = memcmp((void *)&device_prop.uuid, vkDeviceUUID, UUID_SIZE);
                if (ret == 0)
                {
                    CODE_API::CW_GetDeviceProperties(&device_prop, current_device);
                    break;
                }
            }
            else
            {
                devices_prohibited++;
            }
            current_device++;
        }
        if (devices_prohibited == device_count)
        {
            CODECUDA_PRINTLN("CUDA error: No Vulkan-CUDA Interop capable GPU found.");
            return C_Res::ERR;
        }

        CODE_API::CW_SetDevice(current_device);
        CODE_API::CW_StreamCreate(&this->stream);
        this->initialized = true;
        CODECUDA_PRINTLN("Initialized: CodeCudaEngine");
        CODECUDA_PRINTLN("GPU Device index: ", current_device, "\nName: ", device_prop.name,
                         " \nWith compute capability: ", device_prop.major, device_prop.minor);
        return C_Res::OK;
    }


    C_Res CodeCudaContext::C_ImportExternalBuffer(HANDLE win_handle, size_t buffer_size)
    {
        cudaExternalMemoryHandleDesc cuda_external_memory_buffer_desc = {};

        cuda_external_memory_buffer_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        cuda_external_memory_buffer_desc.handle.win32.handle = win_handle;
        cuda_external_memory_buffer_desc.size = buffer_size;

        cudaExternalMemory_t cuda_external_memory = nullptr;

        CODE_API::CW_ImportExternalMemory(&cuda_external_memory, &cuda_external_memory_buffer_desc);

        cudaExternalMemoryBufferDesc buffer_desc = {};
        buffer_desc.offset = 0;
        buffer_desc.size = buffer_size;

        CODE_API::CW_ExternalMemoryGetMappedBuffer(&this->mappedPtr, cuda_external_memory, &buffer_desc);

        this->cpu_launcher.task = [this]()
        {
            C_UpdateSim();
            if (grid_div_d == nullptr)
            {
                CODE_API::CW_Malloc(&grid_pressures_d, simulation.grid.size() * sizeof(float));
                CODE_API::CW_Malloc(&grid_div_d, simulation.grid.size() * sizeof(float));
                CODE_API::CW_Malloc(&grid_v_d, simulation.grid.size() * sizeof(float));
                CODE_API::CW_Malloc(&u_d, simulation.u_edges.size() * sizeof(float));
                CODE_API::CW_Malloc(&v_d, simulation.v_edges.size() * sizeof(float));
            }
            std::vector<float> divs;
            std::vector<float> v;
            std::vector<float> pressures;
            std::vector<float> u_edges;
            std::vector<float> v_edges;
            divs.resize(simulation.grid.size());
            v.resize(simulation.grid.size());
            pressures.resize(simulation.grid.size());
            u_edges.resize(simulation.u_edges.size());
            v_edges.resize(simulation.v_edges.size());
            for (int i = 0; i < simulation.grid.size(); ++i)
            {
                divs[i] = simulation.grid[i].div;
                v[i] = simulation.grid[i].cell_velocity;
                pressures[i] = simulation.grid[i].pressure;
            }
            for (int i = 0; i < simulation.u_edges.size(); ++i)
            {
                u_edges[i] = simulation.u_edges[i].vec;
                v_edges[i] = simulation.v_edges[i].vec;
            }


            CODE_API::CW_Memcpy(grid_div_d, divs.data(), simulation.grid.size() * sizeof(float),
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(grid_pressures_d, v.data(), simulation.grid.size() * sizeof(float),
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(grid_v_d, pressures.data(), simulation.grid.size() * sizeof(float),
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(u_d, u_edges.data(), simulation.u_edges.size() * sizeof(float), cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(v_d, v_edges.data(), simulation.v_edges.size() * sizeof(float), cudaMemcpyHostToDevice);

            this->curr_t += time_step;
        };
        this->kernel_launcher.kernel = [this](cudaStream_t stream)
        {
            dim3 grid((1024 * 1024) / 128, 1, 1);
            dim3 block(128, 1, 1);
            code_kernels::code_tests::k_simulation_read<<<grid, block, 0, stream>>>(
                1024 * 1024, simulation.w, simulation.h, (float *)u_d, (float *)v_d, (float *)grid_v_d,
                (float *)grid_div_d, (float *)grid_pressures_d, (float *)this->mappedPtr);
        };
        CODE_API::CW_DeviceSynchronize();
        return C_Res::OK;
    }

    C_Res CodeCudaContext::C_ImportExternalSemaphore(HANDLE win_handle)
    {
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        desc.handle.win32.handle = win_handle;

        CODE_API::CW_ImportExternalSemaphore(&this->external_semaphore, &desc);

        return C_Res::OK;
    }
    C_Res CodeCudaContext::C_Shutdown()
    {
        CODECUDA_PRINTLN("Shutdown: CodeCudaEngine");
        if (!this->initialized)
            return C_Res::ERR;

        CODE_API::CW_StreamSynchronize(this->stream);
        CODE_API::CW_StreamDestroy(this->stream);

        CODE_API::CW_Free(grid_pressures_d);
        CODE_API::CW_Free(grid_div_d);
        CODE_API::CW_Free(grid_v_d);
        this->stream = nullptr;
        this->device = -1;
        this->initialized = false;
        return C_Res::OK;
    }

    C_Res C_AddVelocity(int x_pos, int y_pos, int radius, float vel_x, float vel_y)
    {
        simulation.AddVelocity(x_pos, y_pos, radius, vel_x, vel_y);
        return C_Res::OK;
    }

    C_Res C_AddRadialVelocity(int x_pos, int y_pos, int radius, float scale)
    {
        simulation.AddRadialVelocity(x_pos, y_pos, radius, scale);
        return C_Res::OK;
    }
    C_Res C_AddVelocity()
    {
        int x = rand() % simulation.w;
        int y = rand() % simulation.h;

        int r = rand() % 50;
        int scale = rand() % 10;

        float vel_x = (float(rand() % 50) / 50.0f) * float(scale);
        float vel_y = (float(rand() % 50) / 50.0f) * float(scale);

        if (simulation.sim_step_idx % 2 == 0)
        {
            simulation.AddVelocity(x, y, r, vel_x, vel_y);
        }
        return C_Res::OK;
    }
    C_Res C_UpdateSim()
    {
        if (!simulation.ready_to_run)
        {
            simulation.InitGrid(50, 50);
        }
        simulation.UpdateSimulation();
        return C_Res::OK;
    }
    C_Res C_Matmul(CodeCudaContext *code_cuda_context, const int M, const int N, const int K, const float *a,
                   const float *b, float *c)
    {
        if (c == nullptr)
        {
            CODECUDA_LOG_WARNING("target buffer is empty");
            return C_Res::ERR;
        }
        float *d_A, *d_B, *d_C;
        CODE_API::CW_Malloc(&d_A, M * K * sizeof(float));
        CODE_API::CW_Malloc(&d_B, K * N * sizeof(float));
        CODE_API::CW_Malloc(&d_C, M * N * sizeof(float));

        CODE_API::CW_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
        CODE_API::CW_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

        constexpr uint32_t BM = k_auto_tunning_params::BM;
        constexpr uint32_t BK = k_auto_tunning_params::BK;
        constexpr uint32_t BN = k_auto_tunning_params::BN;
        constexpr uint32_t BSIZE = k_auto_tunning_params::BSIZE;

        dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
        dim3 block(BSIZE);
        code_kernels::code_math::
            k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float), code_cuda_context->stream>>>(
                M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);

        CODE_API::CW_GetLastError();
        CODE_API::CW_DeviceSynchronize();

        CODE_API::CW_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        CODE_API::CW_Free(d_A);
        CODE_API::CW_Free(d_B);
        CODE_API::CW_Free(d_C);

        return C_Res::OK;
    }

    namespace CodeBenchmarking
    {

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
        void C_Matmul_Test(CodeCudaContext *code_cuda_context, const int M, const int N, const int K, const float *a,
                           const float *b, float *c, int runs)
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
            CODE_API::CW_Malloc(&d_A, M * K * sizeof(float));
            CODE_API::CW_Malloc(&d_B, K * N * sizeof(float));
            CODE_API::CW_Malloc(&d_C, M * N * sizeof(float));

            CODE_API::CW_Memcpy(d_A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(d_B, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            CODE_API::CW_EventCreate(&start);
            CODE_API::CW_EventCreate(&stop);
            /*
            dim3 grid(ceil(double(N)/32.0f), ceil(double(M)/32.0f));
            dim3 block(32, 32);
            */

            std::map<std::string, kernel_launcher> kernels;

            using namespace code_kernels::code_math;
            add_kernel_launcher(
                "naive_coalescent",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    dim3 grid(ceil(double(N) / 32.0f), ceil(double(M) / 32.0f));
                    dim3 block(32, 32);
                    k_matmul_naive<<<grid, block, 0, code_cuda_context->stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);
            add_kernel_launcher(
                "smem",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    dim3 grid(ceil(double(M) / 32.0f), ceil(double(N) / 32.0f));
                    dim3 block(32 * 32);
                    k_matmul_x_y<<<grid, block, block.x * sizeof(float) * 2, code_cuda_context->stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_col_tile",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BM * BK);
                    k_matmul_bt_col_tile<<<grid, block, (BN * BK + BK * BM) * sizeof(float),
                                           code_cuda_context->stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_row_tile",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(M) / double(BM)), ceil(double(N) / double(BK * BK)));
                    dim3 block(BM * BK);
                    k_matmul_bt_row_tile<<<grid, block, (BM * BK + BK * BN) * sizeof(float),
                                           code_cuda_context->stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_2d",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    k_matmul_bt_2d_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float),
                                             code_cuda_context->stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "b_tilling_2d_transposed",
                [N, M, K, d_A, d_B, d_C](cudaStream_t stream)
                {
                    constexpr uint32_t BM = 64;
                    constexpr uint32_t BK = 8;
                    constexpr uint32_t BN = 64;
                    dim3 grid(ceil(double(N) / double(BK * BK)), ceil(double(M) / double(BK * BK)));
                    dim3 block(BK * BK);
                    k_matmul_bt_2d_tilling_transposed_a<<<grid, block, (BM * BK + BK * BN) * sizeof(float), stream>>>(
                        M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);

            add_kernel_launcher(
                "warp_tilling",
                [N, M, K, d_A, d_B, d_C, code_cuda_context](cudaStream_t)
                {
                    constexpr uint32_t BM = k_auto_tunning_params::BM;
                    constexpr uint32_t BK = k_auto_tunning_params::BK;
                    constexpr uint32_t BN = k_auto_tunning_params::BN;
                    constexpr uint32_t BSIZE = k_auto_tunning_params::BSIZE;

                    dim3 grid(ceil(double(N) / double(BN)), ceil(double(M) / double(BM)));
                    dim3 block(BSIZE);
                    k_matmul_bt_warp_tilling<<<grid, block, (BM * BK + BK * BN) * sizeof(float),
                                               code_cuda_context->stream>>>(M, N, K, 1.0f, 0.0f, d_A, d_B, d_C);
                },
                kernels);


            for (int i = 0; i < 5; ++i)
            {
                kernels.at("warp_tilling").kernel(code_cuda_context->stream);
            }
            CODE_API::CW_GetLastError();
            CODE_API::CW_DeviceSynchronize();

            CODE_API::CW_EventRecord(start);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("warp_tilling").kernel(code_cuda_context->stream);
            }
            CODE_API::CW_GetLastError();

            CODE_API::CW_EventRecord(stop);
            CODE_API::CW_EventSynchronize(stop);
            float ms = 0;
            CODE_API::CW_EventElapsedTime(&ms, start, stop);
            double ms_real = ms / double(runs);
            CODECUDA_PRINTLN("Average Time personal (ms): ", ms_real);
            auto flops = int64_t(2 * int64_t(M) * int64_t(N) * int64_t(K));

            double gflops = (double(flops) * 1.0e-9f) / double(ms_real / 1000.0f);
            CODECUDA_PRINTLN("GFLOPS/s personal: ", gflops);

            // cubulas
            float *d_A_cublas, *d_B_cublas, *d_C_cublas;
            CODE_API::CW_Malloc(&d_A_cublas, M * K * sizeof(float));
            CODE_API::CW_Malloc(&d_B_cublas, K * N * sizeof(float));
            CODE_API::CW_Malloc(&d_C_cublas, M * N * sizeof(float));

            CODE_API::CW_Memcpy(d_A_cublas, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(d_B_cublas, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

            float alpha = 1.0f;
            float beta = 0.0f;
            cublasHandle_t handle;
            cudaEvent_t start_cublas, stop_cublas;
            CODE_API::CW_EventCreate(&start_cublas);
            CODE_API::CW_EventCreate(&stop_cublas);

            CUBLAS_CHECK(cublasCreate_v2(&handle));

            add_kernel_launcher(
                "cublas",
                [handle, N, M, K, alpha, d_B_cublas, d_A_cublas, beta, d_C_cublas](cudaStream_t)
                {
                    CUBLAS_CHECK(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_cublas, N,
                                                d_A_cublas, K, &beta, d_C_cublas, N));
                },
                kernels);

            for (int i = 0; i < 5; ++i)
            {
                kernels.at("cublas").kernel(code_cuda_context->stream);
            }
            CODE_API::CW_EventRecord(start_cublas);
            for (int i = 0; i < runs; ++i)
            {
                kernels.at("cublas").kernel(code_cuda_context->stream);
            }
            CODE_API::CW_EventRecord(stop_cublas);
            CODE_API::CW_EventSynchronize(stop_cublas);
            float ms_cublas = 0;
            CODE_API::CW_EventElapsedTime(&ms_cublas, start_cublas, stop_cublas);
            double ms_real_cublas = ms_cublas / double(runs);
            CODECUDA_PRINTLN("Average Time cublas (ms): ", ms_real_cublas);
            double gflops_cublas = double(double(flops) * 1.0e-9) / double(ms_real_cublas / 1000.0);
            CODECUDA_PRINTLN("GFLOPS/s cublas: ", gflops_cublas);
            // personal vs cublas
            float *d_C_err;
            CODE_API::CW_Malloc(&d_C_err, M * N * sizeof(float));

            dim3 block_err(128, 1, 1);
            dim3 grid_err(ceil(double(M * N) / 128.0), 1, 1);
            k_check_mat_err<<<grid_err, block_err, 0, code_cuda_context->stream>>>(M, N, d_C, d_C_cublas, d_C_err);
            CODE_API::CW_GetLastError();
            CODE_API::CW_DeviceSynchronize();

            float *errMatrix;
            errMatrix = new float[M * N];
            CODE_API::CW_Memcpy(errMatrix, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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

            std::string errOutput;
            errOutput += "----- Matmul compare -----\n";
            errOutput += std::format("Average error (personal vs cublas): {:.6f}\n", err_average);
            errOutput += std::format("Max error(personal vs cublas): {:.6f}\n", max_error);

            if (M * N < 8192)
            {
                // personal vs cpu
                auto *h_C = (float *)malloc(M * N * sizeof(float));
                cpu_matmul(M, N, K, a, b, h_C);
                float *d_h_C;
                CODE_API::CW_Malloc(&d_h_C, M * N * sizeof(float));

                CODE_API::CW_Memcpy(d_h_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

                k_check_mat_err<<<grid_err, block_err, 0, code_cuda_context->stream>>>(M, N, d_C, d_h_C, d_C_err);
                CODE_API::CW_GetLastError();
                CODE_API::CW_DeviceSynchronize();
                float *errMatrix_cpu;
                errMatrix_cpu = new float[M * N];
                CODE_API::CW_Memcpy(errMatrix_cpu, d_C_err, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
                CODE_API::CW_Free(d_h_C);
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

            CODE_API::CW_Memcpy(c, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            CODE_API::CW_Free(d_A);
            CODE_API::CW_Free(d_B);
            CODE_API::CW_Free(d_C);
            CODE_API::CW_Free(d_A_cublas);
            CODE_API::CW_Free(d_B_cublas);
            CODE_API::CW_Free(d_C_cublas);
            CODE_API::CW_Free(d_C_err);
            CODE_API::CW_EventDestroy(start);
            CODE_API::CW_EventDestroy(stop);
            CODE_API::CW_EventDestroy(start_cublas);
            CODE_API::CW_EventDestroy(stop_cublas);
            CUBLAS_CHECK(cublasDestroy_v2(handle));
            delete[] errMatrix;
            cublasShutdown();
        }
    } // namespace CodeBenchmarking


} // namespace CodeCuda


#endif // CODECUDA_CU
