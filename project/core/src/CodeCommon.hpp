//
// Created by carlo on 2026-02-01.
//

#ifndef CODECOMMON_HPP
#define CODECOMMON_HPP


#include <algorithm>
#include "assert.h"
#include "common/Logger.hpp"

namespace CodeCommon
{
#define CUDA_CHECK(x)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = x;                                                                                           \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            CODECUDA_LOG_ERROR("CUDA error: ", cudaGetErrorString(err));                                               \
            assert(false);                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

#define CUBLAS_CHECK(x)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t err = x;                                                                                        \
        if (err != CUBLAS_STATUS_SUCCESS)                                                                              \
        {                                                                                                              \
            CODECUDA_LOG_ERROR("CUBLAS error code: ", static_cast<int>(err));                                          \
            assert(false);                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

} // namespace CodeCommon


namespace CODE_API
{
    inline void CW_Free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }
    inline void CW_DeviceSynchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }
    inline void CW_GetLastError() { CUDA_CHECK(cudaGetLastError()); }
    inline void CW_PeekAtLastError() { CUDA_CHECK(cudaPeekAtLastError()); }
    inline void CW_FreeZero() { CUDA_CHECK(cudaFree(nullptr)); }
    inline void CW_ImportExternalMemory(cudaExternalMemory_t *extMem_out,
                                        const cudaExternalMemoryHandleDesc *memHandleDesc)
    {
        CUDA_CHECK(cudaImportExternalMemory(extMem_out, memHandleDesc));
    }

    inline void CW_ExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem,
                                                 const struct cudaExternalMemoryBufferDesc *bufferDesc)
    {
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc));
    }
    inline void CW_StreamCreate(cudaStream_t *stream) { CUDA_CHECK(cudaStreamCreate(stream)); }
    inline void CW_StreamDestroy(cudaStream_t stream) { CUDA_CHECK(cudaStreamDestroy(stream)); }
    inline void CW_StreamSynchronize(cudaStream_t stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    inline void CW_EventCreate(cudaEvent_t *event) { CUDA_CHECK(cudaEventCreate(event)); }
    inline void CW_EventDestroy(cudaEvent_t event) { CUDA_CHECK(cudaEventDestroy(event)); }
    inline void CW_EventRecord(cudaEvent_t event, cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
    inline void CW_EventSynchronize(cudaEvent_t event) { CUDA_CHECK(cudaEventSynchronize(event)); }
    inline void CW_EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)
    {
        CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
    }
    inline void CW_Memset(void *dst, int value, size_t count) { CUDA_CHECK(cudaMemset(dst, value, count)); }
    inline void CW_SetDevice(int device) { CUDA_CHECK(cudaSetDevice(device)); }
    inline void CW_GetDevice(int *device) { CUDA_CHECK(cudaGetDevice(device)); }
    inline void CW_GetDeviceCount(int *device) { CUDA_CHECK(cudaGetDeviceCount(device)); }
    inline void CW_GetDeviceProperties(cudaDeviceProp *prop, int device)
    {
        CUDA_CHECK(cudaGetDeviceProperties(prop, device));
    }
    template <class T>
    inline void CW_HostMalloc(T **ptr, size_t size, unsigned int flags = cudaHostAllocDefault)
    {
        CUDA_CHECK(cudaHostAlloc((void **)ptr, size, flags));
    }
    inline void CW_HostFree(void *ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
    template <class T>
    inline void CW_MallocManaged(T **ptr, size_t size, unsigned int flags = cudaMemAttachGlobal)
    {
        CUDA_CHECK(cudaMallocManaged((void **)ptr, size, flags));
    }
    template <class T>
    inline void CW_Malloc(T **ptr, size_t size)
    {
        CUDA_CHECK(cudaMalloc((void **)ptr, size));
    }
    inline void CW_Memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    {
        CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
    }
    inline void CW_MemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, count, kind, stream));
    }
    inline void CW_DeviceReset() { CUDA_CHECK(cudaDeviceReset()); }
    inline void CW_DestroyExternalMemory(cudaExternalMemory_t extMem) { CUDA_CHECK(cudaDestroyExternalMemory(extMem)); }
    inline void CW_ImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out,
                                           const cudaExternalSemaphoreHandleDesc *semHandleDesc)
    {
        CUDA_CHECK(cudaImportExternalSemaphore(extSem_out, semHandleDesc));
    }

    inline void CW_DestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
    {
        CUDA_CHECK(cudaDestroyExternalSemaphore(extSem));
    }

    inline void CW_WaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray,
                                               const cudaExternalSemaphoreWaitParams *paramsArray,
                                               unsigned int numExtSems, cudaStream_t stream)
    {
        CUDA_CHECK(cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream));
    }

    inline void CW_SignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray,
                                                 const cudaExternalSemaphoreSignalParams *paramsArray,
                                                 unsigned int numExtSems, cudaStream_t stream)
    {
        CUDA_CHECK(cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream));
    }
    inline void CW_StreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(stream, flags));
    }

    inline void CW_StreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority)
    {
        CUDA_CHECK(cudaStreamCreateWithPriority(stream, flags, priority));
    }

    inline void CW_StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0)
    {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, flags));
    }

    inline void CW_StreamQuery(cudaStream_t stream) { CUDA_CHECK(cudaStreamQuery(stream)); }
    inline void CW_EventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(event, flags));
    }

    inline void CW_EventQuery(cudaEvent_t event) { CUDA_CHECK(cudaEventQuery(event)); }
    inline void CW_MemsetAsync(void *dst, int value, size_t count, cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemsetAsync(dst, value, count, stream));
    }
    inline void CW_Memcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                                 cudaMemcpyKind kind, cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream));
    }
    template <class T>
    inline void CW_MallocPitch(T **ptr, size_t *pitch, size_t widthBytes, size_t height)
    {
        CUDA_CHECK(cudaMallocPitch((void **)ptr, pitch, widthBytes, height));
    }
    inline void CW_MemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaMemPrefetchAsync(devPtr, count, dstDevice, stream));
    }
    inline void CW_DeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)
    {
        CUDA_CHECK(cudaDeviceGetAttribute(value, attr, device));
    }
    inline void CW_CheckKernelLaunch() { CUDA_CHECK(cudaPeekAtLastError()); }
} // namespace CODE_API


struct k_auto_tunning_params
{
    static constexpr uint32_t WSIZE = 32;
    static constexpr uint32_t BN = 128;
    static constexpr uint32_t BM = 64;
    static constexpr uint32_t BK = 16;
    static constexpr uint32_t WN = 64;
    static constexpr uint32_t WM = 32;
    // this is the total block size calculated based on BM, WM... so
    static constexpr uint32_t BSIZE = (BM / WM) * (BN / WN) * WSIZE;
    static constexpr uint32_t WCOLS = BN / WN;
    static constexpr uint32_t WROWS = BM / WM;
    static constexpr uint32_t WNITER = 2;

    static constexpr uint32_t TN = 4;
    static constexpr uint32_t TM = 4;

    static constexpr uint32_t WMITER = (WM * WN) / (WSIZE * TM * TN * WNITER);
    static constexpr uint32_t WSUBN = WN / WNITER;
    static constexpr uint32_t WSUBM = WM / WMITER;
    static constexpr uint32_t WTCOLS = WSUBN / TN;
    static constexpr uint32_t WTROWS = WSIZE / WTCOLS;
};

namespace CodeBenchmarking
{

    struct c_matmul_benchmark_result
    {
        int32_t M = 0;
        int32_t N = 0;
        int32_t K = 0;
        int32_t runs = 0;
        double personal_ms = 0.0;
        double personal_gflops = 0.0;
        double cublas_ms = 0.0;
        double cublas_gflops = 0.0;
        double average_error = 0.0;
        double max_error = 0.0;
        bool passed = false;
    };

    std::string BuildMatmulBenchmarkResultJson(const c_matmul_benchmark_result &result)
    {
        using Params = k_auto_tunning_params;

        std::ostringstream output;
        output << std::boolalpha;
        output << "{\n";
        output << "  \"shape\": {\"M\": " << result.M << ", \"N\": " << result.N << ", \"K\": " << result.K << "},\n";
        output << "  \"runs\": " << result.runs << ",\n";
        output << "  \"autotuning_params\": {\n";
        output << "    \"WSIZE\": " << Params::WSIZE << ",\n";
        output << "    \"BN\": " << Params::BN << ",\n";
        output << "    \"BM\": " << Params::BM << ",\n";
        output << "    \"BK\": " << Params::BK << ",\n";
        output << "    \"WN\": " << Params::WN << ",\n";
        output << "    \"WM\": " << Params::WM << ",\n";
        output << "    \"BSIZE\": " << Params::BSIZE << ",\n";
        output << "    \"WCOLS\": " << Params::WCOLS << ",\n";
        output << "    \"WROWS\": " << Params::WROWS << ",\n";
        output << "    \"WNITER\": " << Params::WNITER << ",\n";
        output << "    \"TN\": " << Params::TN << ",\n";
        output << "    \"TM\": " << Params::TM << ",\n";
        output << "    \"WMITER\": " << Params::WMITER << ",\n";
        output << "    \"WSUBN\": " << Params::WSUBN << ",\n";
        output << "    \"WSUBM\": " << Params::WSUBM << ",\n";
        output << "    \"WTCOLS\": " << Params::WTCOLS << ",\n";
        output << "    \"WTROWS\": " << Params::WTROWS << "\n";
        output << "  },\n";
        output << "  \"personal\": {\"kernel\": \"warp_tilling\", \"ms\": " << result.personal_ms
               << ", \"gflops\": " << result.personal_gflops << "},\n";
        output << "  \"cublas\": {\"ms\": " << result.cublas_ms << ", \"gflops\": " << result.cublas_gflops << "},\n";
        output << "  \"accuracy\": {\"average_error\": " << result.average_error
               << ", \"max_error\": " << result.max_error << "},\n";
        output << "  \"passed\": " << result.passed << "\n";
        output << "}";
        return output.str();
    }

    std::string TrimTrailingWhitespace(std::string text)
    {
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        {
            text.pop_back();
        }
        return text;
    }

    static void C_SaveMatmulBenchmarkResultJson(const char *path, const c_matmul_benchmark_result &result)
    {
        if (path == nullptr || path[0] == '\0')
        {
            CODECUDA_LOG_WARNING("benchmark json path is empty");
            return;
        }

        const std::string result_json = BuildMatmulBenchmarkResultJson(result);

        std::ifstream existing_input(path);
        std::string existing;
        if (existing_input)
        {
            existing.assign(std::istreambuf_iterator<char>(existing_input), std::istreambuf_iterator<char>());
        }

        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output)
        {
            CODECUDA_LOG_WARNING("failed to open benchmark json path: ", path);
            return;
        }

        existing = TrimTrailingWhitespace(existing);
        if (existing.size() >= 2 && existing.front() == '[' && existing.back() == ']')
        {
            existing.pop_back();
            existing = TrimTrailingWhitespace(existing);
            output << existing;
            if (existing.size() > 1)
            {
                output << ",\n";
            }
            output << result_json << "\n]\n";
            return;
        }

        output << "[\n" << result_json << "\n]\n";
    }


} // namespace CodeBenchmarking
namespace CodeSimulation
{
    struct c_edge
    {
        float vec = 0.0f;
        bool is_wall = false;
        float GetState() { return is_wall ? 0.0f : 1.0f; }
    };
    struct c_cell
    {
        float div = 0.0f;
        bool is_wall = false;
        int s = 0;
        float density = 1.0f;
        float pressure = 0.0f;
        float cell_velocity = 0.0f;
    };
    struct c_grid
    {
        c_grid() = default;
        void InitGrid(int width, int height)
        {
            this->edge_w = (width + 1);
            this->edge_h = (height + 1);
            this->w = width;
            this->h = height;
            this->dx = 1.0f / float(w);
            this->dy = 1.0f / float(h);
            grid.reserve(width * height);
            u_edges.reserve(edge_w * edge_h);
            v_edges.reserve(edge_w * edge_h);

            for (int i = 0; i < u_edges.capacity(); ++i)
            {
                u_edges.emplace_back(c_edge{});
                int x = i % edge_w;
                int y = i / edge_w;
                u_edges[i].is_wall = x == 0 || x == edge_w - 1 || y == 0 || y == edge_h - 1;
            }
            for (int i = 0; i < v_edges.capacity(); ++i)
            {
                v_edges.emplace_back(c_edge{});
                int x = i % edge_w;
                int y = i / edge_w;
                v_edges[i].is_wall = x == 0 || x == edge_w - 1 || y == 0 || y == edge_h - 1;
                // int v = i % 2 == 0 ? -1 : 1;
                v_edges[i].vec = float(rand() % 1000) / 1000.0f;
            }

            for (int i = 0; i < grid.capacity(); ++i)
            {
                grid.emplace_back(c_cell{});
                int x = i % w;
                int y = i / w;
                grid[i].is_wall = x == 0 || x == w - 1 || y == 0 || y == h - 1;
                if (!grid[i].is_wall)
                {
                    valid_cell_count++;
                }
            }
            for (int i = 0; i < grid.size(); ++i)
            {
                c_edge *edge_u_left_out = nullptr;
                c_edge *edge_u_right_out = nullptr;
                c_edge *edge_v_top_out = nullptr;
                c_edge *edge_v_bottom_out = nullptr;
                int x = i % w;
                int y = i / w;
                GetCellEdges(x, y, edge_u_left_out, edge_u_right_out, edge_v_top_out, edge_v_bottom_out);
                grid[i].s = edge_u_left_out->GetState() + edge_u_right_out->GetState() + edge_v_top_out->GetState() +
                    edge_v_bottom_out->GetState();
            }
            ready_to_run = true;
        }
        void AddRadialVelocity(int x_pos, int y_pos, int radius, float scale)
        {
            if (x_pos < 0 || x_pos >= edge_w || y_pos < 0 || y_pos >= edge_h || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid radial velocity parameters");
                return;
            }

            const int radius_sq = radius * radius;

            for (int y = -radius; y <= radius; ++y)
            {
                for (int x = -radius; x <= radius; ++x)
                {
                    const int x_final = x_pos + x;
                    const int y_final = y_pos + y;

                    if (x_final < 0 || x_final >= edge_w || y_final < 0 || y_final >= edge_h)
                    {
                        continue;
                    }

                    // x and y are already offsets from the center.
                    const int sq_dist = x * x + y * y;

                    if (sq_dist == 0 || sq_dist > radius_sq)
                    {
                        continue;
                    }

                    const int idx = y_final * edge_w + x_final;

                    if (u_edges[idx].is_wall)
                    {
                        continue;
                    }

                    const float inverse_length = 1.0f / std::sqrt(static_cast<float>(sq_dist));

                    const float u = static_cast<float>(x) * inverse_length * scale;
                    const float v = static_cast<float>(y) * inverse_length * scale;

                    u_edges[idx].vec += u;
                    v_edges[idx].vec += v;
                }
            }
        }
        void AddVelocity(int x_pos, int y_pos, int radius, float vel_x, float vel_y)
        {
            if (x_pos < 0 || x_pos >= edge_w || y_pos < 0 || y_pos >= edge_h)
            {
                CODECUDA_PRINTLN("Invalid x,y pos");
                return;
            }
            for (int y = -radius; y < radius; ++y)
            {
                for (int x = -radius; x < radius; ++x)
                {
                    int x_final = x + x_pos;
                    int y_final = y + y_pos;
                    if (x_final < 0 || x_final >= edge_w || y_final < 0 || y_final >= edge_h)
                    {
                        continue;
                    }
                    int sq_dist = pow(x_final - x_pos, 2.0f) + pow(y_final - y_pos, 2.0f);
                    if (sq_dist >= radius * radius)
                    {
                        continue;
                    }
                    int idx = y_final * edge_w + x_final;
                    if (u_edges[idx].is_wall)
                    {
                        continue;
                    }
                    u_edges[idx].vec += vel_x;
                    v_edges[idx].vec += vel_y;
                }
            }
        }
        void RunSimulation(int steps)
        {
            solved_grid_states.resize(steps);
            solved_grid_u.resize(steps);
            solved_grid_v.resize(steps);
            for (int i = 0; i < solved_grid_states.size(); ++i)
            {
                solved_grid_states[i].reserve(grid.size());
            }
            for (int i = 0; i < solved_grid_u.size(); ++i)
            {
                solved_grid_u[i].reserve(u_edges.size());
            }
            for (int i = 0; i < solved_grid_v.size(); ++i)
            {
                solved_grid_v[i].reserve(v_edges.size());
            }

            for (int i = 0; i < steps; ++i)
            {
                UpdateSimulation();
                memcpy(solved_grid_states[i].data(), grid.data(), grid.size() * sizeof(c_cell));
                memcpy(solved_grid_u[i].data(), u_edges.data(), u_edges.size() * sizeof(c_edge));
                memcpy(solved_grid_v[i].data(), v_edges.data(), v_edges.size() * sizeof(c_edge));
            }
        }
        void UpdateSimulation()
        {
            UpdateVelocity();
            UpdateDiv();
            // AdvectVelocity();
            UpdateData();
        }

    private:
        void AdvectVelocity()
        {
            for (int i = 0; i < u_edges.size(); ++i)
            {

                if (u_edges[i].is_wall)
                {
                    continue;
                }
                float grad = u_edges[i].vec - u_edges[i - 1].vec;
                u_edges[i].vec = u_edges[i].vec - ((u_edges[i].vec * dt) * grad) / dx;
            }

            for (int i = 0; i < v_edges.size(); ++i)
            {
                if (v_edges[i].is_wall)
                {
                    continue;
                }
                float grad = v_edges[i].vec - v_edges[i - 1].vec;
                v_edges[i].vec = v_edges[i].vec - ((v_edges[i].vec * dt) * grad) / dy;
            }
        }
        void UpdateData()
        {
            sim_step_idx++;
            total_t += dt;
        }
        void UpdateVelocity()
        {
            // float gravity = (g * 1);
            // if (gravity_sign == 1)
            // {
            //     gravity = g * -1;
            // }
            //
            // for (int i = 0; i < v_edges.size(); ++i)
            // {
            //     if (v_edges[i].is_wall)
            //     {
            //         continue;
            //     }
            //     v_edges[i].vec += (dt * (gravity));
            // }
        }
        void UpdateDiv()
        {
            int converged = 0;
            int idx = 0;
            for (int i = 0; i < grid.size(); ++i)
            {
                if (grid[i].is_wall)
                {
                    continue;
                }
                grid[i].pressure = 0.0f;
            }
            for (int iter = 0; iter < 5; ++iter)
            {
                for (int i = 0; i < grid.size(); ++i)
                {
                    if (grid[i].is_wall)
                    {
                        continue;
                    }
                    int s = grid[i].s;
                    if (s == 0)
                    {
                        CODECUDA_PRINTLN("solid");
                        continue;
                    }
                    c_edge *edge_u_left_out = nullptr;
                    c_edge *edge_u_right_out = nullptr;
                    c_edge *edge_v_top_out = nullptr;
                    c_edge *edge_v_bottom_out = nullptr;
                    int x = i % w;
                    int y = i / w;
                    GetCellEdges(x, y, edge_u_left_out, edge_u_right_out, edge_v_top_out, edge_v_bottom_out);
                    grid[i].div = Overrelaxation(edge_u_right_out->vec - edge_u_left_out->vec + edge_v_top_out->vec -
                                                 edge_v_bottom_out->vec);

                    float correction = grid[i].div / float(s);

                    edge_u_left_out->vec += correction * edge_u_left_out->GetState();
                    edge_u_right_out->vec -= correction * edge_u_right_out->GetState();
                    edge_v_bottom_out->vec += correction * edge_v_bottom_out->GetState();
                    edge_v_top_out->vec -= correction * edge_v_top_out->GetState();
                    grid[i].pressure += correction * (grid[i].density * dx / dt);

                    grid[i].cell_velocity =
                        (edge_u_left_out->vec + edge_u_right_out->vec + edge_v_bottom_out->vec + edge_v_top_out->vec) /
                        4.0f;
                }

                converged = 0;
                for (int i = 0; i < grid.size(); ++i)
                {
                    if (grid[i].is_wall)
                    {
                        continue;
                    }
                    c_edge *edge_u_left_out = nullptr;
                    c_edge *edge_u_right_out = nullptr;
                    c_edge *edge_v_top_out = nullptr;
                    c_edge *edge_v_bottom_out = nullptr;
                    int x = i % w;
                    int y = i / w;
                    GetCellEdges(x, y, edge_u_left_out, edge_u_right_out, edge_v_top_out, edge_v_bottom_out);
                    grid[i].div =
                        edge_u_right_out->vec - edge_u_left_out->vec + edge_v_top_out->vec - edge_v_bottom_out->vec;
                    if (std::abs(grid[i].div) < epsilon)
                    {
                        converged++;
                    };
                }
                idx++;
            }
            PrintDivergenceConvergence(idx);
        }


        void GetCellEdges(int x, int y, c_edge *&edge_u_left_out, c_edge *&edge_u_right_out, c_edge *&edge_v_top_out,
                          c_edge *&edge_v_bottom_out)
        {

            edge_u_left_out = &u_edges[y * edge_w + x];
            edge_u_right_out = &u_edges[y * edge_w + (x + 1)];

            edge_v_top_out = &v_edges[y * edge_w + x];
            edge_v_bottom_out = &v_edges[(y + 1) * edge_w + x];
        }
        float Overrelaxation(float div) { return div * 1.9f; }
        void PrintDivergenceConvergence(int iteration)
        {
            int totalCells = 0;
            int convergedCells = 0;

            float sumAbsDiv = 0.0f;
            float maxAbsDiv = 0.0f;

            float sumAbsPres = 0.0f;
            float maxAbsPres = 0.0f;

            float sumAbsU = 0.0f;
            float maxAbsU = 0.0f;
            float minU = std::numeric_limits<float>::max();
            float maxU = std::numeric_limits<float>::lowest();
            int validUCount = 0;

            float sumAbsV = 0.0f;
            float maxAbsV = 0.0f;
            float minV = std::numeric_limits<float>::max();
            float maxV = std::numeric_limits<float>::lowest();
            int validVCount = 0;

            for (int i = 0; i < grid.size(); ++i)
            {
                if (grid[i].is_wall)
                {
                    continue;
                }

                const float absDiv = std::abs(grid[i].div);
                const float absPres = std::abs(grid[i].pressure);

                totalCells++;

                sumAbsDiv += absDiv;
                maxAbsDiv = std::max(maxAbsDiv, absDiv);

                sumAbsPres += absPres;
                maxAbsPres = std::max(maxAbsPres, absPres);

                if (absDiv < epsilon)
                {
                    convergedCells++;
                }
            }

            for (const c_edge &edge : u_edges)
            {
                if (edge.is_wall)
                {
                    continue;
                }

                const float u = edge.vec;
                const float absU = std::abs(u);

                sumAbsU += absU;
                maxAbsU = std::max(maxAbsU, absU);
                minU = std::min(minU, u);
                maxU = std::max(maxU, u);
                validUCount++;
            }

            for (const c_edge &edge : v_edges)
            {
                if (edge.is_wall)
                {
                    continue;
                }

                const float v = edge.vec;
                const float absV = std::abs(v);

                sumAbsV += absV;
                maxAbsV = std::max(maxAbsV, absV);
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
                validVCount++;
            }

            const float avgAbsDiv = totalCells > 0 ? sumAbsDiv / static_cast<float>(totalCells) : 0.0f;

            const float avgAbsPres = totalCells > 0 ? sumAbsPres / static_cast<float>(totalCells) : 0.0f;

            const float avgAbsU = validUCount > 0 ? sumAbsU / static_cast<float>(validUCount) : 0.0f;

            const float avgAbsV = validVCount > 0 ? sumAbsV / static_cast<float>(validVCount) : 0.0f;

            if (validUCount == 0)
            {
                minU = 0.0f;
                maxU = 0.0f;
            }

            if (validVCount == 0)
            {
                minV = 0.0f;
                maxV = 0.0f;
            }
            std::cout
                << std::setprecision(2)
                << "step=" << sim_step_idx
                << " | time=" << total_t << "s"
                << " | iter=" << iteration
                << " | converged=" << convergedCells << "/" << totalCells
                << " | div(avg/max)=" << avgAbsDiv << "/" << maxAbsDiv
                << " | pressure(avg/max)=" << avgAbsPres << "/" << maxAbsPres
                << " | u(avg/max/range)=" << avgAbsU << "/" << maxAbsU
                << "/[" << minU << "," << maxU << "]"
                << " | v(avg/max/range)=" << avgAbsV << "/" << maxAbsV
                << "/[" << minV << "," << maxV << "]"
                << '\n';
        }
        const float epsilon = 0.0001f;
        float dt = 1.0f / 60.0f;
        float g = -0.1f;
        int valid_cell_count = 0;
        float dx = 0.0f;
        float dy = 0.0f;

    public:
        bool ready_to_run = false;
        int w = -1;
        int h = -1;
        int edge_w = -1;
        int edge_h = -1;
        std::vector<c_cell> grid;
        std::vector<c_edge> u_edges;
        std::vector<c_edge> v_edges;
        std::vector<std::vector<c_cell>> solved_grid_states;
        std::vector<std::vector<c_cell>> solved_grid_u;
        std::vector<std::vector<c_cell>> solved_grid_v;
        int gravity_sign = 1;
        int64_t sim_step_idx = 0;
        float total_t = 0.0f;
    };


} // namespace CodeSimulation
#endif // CODECOMMON_HPP
