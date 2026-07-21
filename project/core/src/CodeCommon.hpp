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


    struct c_cells
    {
        std::vector<float> divs;
        std::vector<float> pressures;
        std::vector<float> smoke;
        std::vector<uint8_t> is_walls;
        std::vector<uint8_t> edges_states_count;
        int valid_cell_count = 0;
        int w = -1;
        int h = -1;

        void Resize(int w, int h)
        {
            this->w = w;
            this->h = h;
            divs.resize(w * h);
            pressures.resize(w * h);
            is_walls.resize(w * h);
            smoke.resize(w * h);
            edges_states_count.resize(w * h);
            valid_cell_count = 0;

            for (int i = 0; i < is_walls.size(); ++i)
            {
                int x = i % w;
                int y = i / w;
                is_walls[i] = IsSolidCell(x, y);
                if (!is_walls[i])
                {
                    valid_cell_count++;
                }
            }
        }
        void Reset()
        {
            this->w = w;
            this->h = h;
            divs.clear();
            pressures.clear();
            is_walls.clear();
            smoke.clear();
            edges_states_count.clear();
            valid_cell_count = 0;
        }
        float GetCellFluidState(int x, int y)
        {
            assert(x >= 0 && x < w);
            assert(y >= 0 && y < h);

            return is_walls[y * w + x] ? 0.0f : 1.0f;
        }
        float &GetCellPressure(int x, int y)
        {
            assert(x < w && x >= 0);
            assert(y < h && y >= 0);
            return pressures[y * w + x];
        }

        uint8_t &GetCellEdgesStateCount(int x, int y)
        {
            assert(x < w && x >= 0);
            assert(y < h && y >= 0);
            return edges_states_count[y * w + x];
        }

        bool IsSolidCell(int x, int y) const
        {
            if (x == 0 || x == w - 1 || y == 0 || y == h - 1)
            {
                return true;
            }

            const float px = (float(x) + 0.5f) / float(w);
            const float py = (float(y) + 0.5f) / float(h);

            const float dx = px - 0.5f;
            const float dy = py - 0.5f;

            constexpr float radius = 0.05f;

            return dx * dx + dy * dy <= radius * radius;
        }
    };
    struct c_edges
    {
        int edges_w;
        int edges_h;
        std::vector<float> u;
        std::vector<float> v;
        std::vector<uint8_t> is_walls_u;
        std::vector<uint8_t> is_walls_v;

        float &GetV(int x, int y) { return v[y * edges_w + x]; }
        float &GetU(int x, int y) { return u[y * edges_w + x]; }

        float GetStateU(int x, int y) { return is_walls_u[y * edges_w + x] == 1 ? 0.0f : 1.0f; }
        float GetStateV(int x, int y) { return is_walls_v[y * edges_w + x] == 1 ? 0.0f : 1.0f; }

        uint8_t &GetWallU(int x, int y) { return is_walls_u[y * edges_w + x]; }

        uint8_t &GetWallV(int x, int y) { return is_walls_v[y * edges_w + x]; }
        void Resize(int w, int h)
        {
            this->edges_w = w;
            this->edges_h = h;
            u.resize(w * h);
            v.resize(w * h);
            is_walls_u.resize(w * h);
            is_walls_v.resize(w * h);

            for (int i = 0; i < u.size(); ++i)
            {
                int x = i % edges_w;
                int y = i / edges_w;
                is_walls_u[i] = IsWall(x, y, edges_w, edges_h);
                u[i] = 0.001f;
            }
            for (int i = 0; i < v.size(); ++i)
            {
                int x = i % edges_w;
                int y = i / edges_w;
                is_walls_v[i] = IsWall(x, y, edges_w, edges_h);
                v[i] = 0.001f;
            }
        }

        void Reset()
        {
            this->edges_w = -1;
            this->edges_h = -1;
            u.clear();
            v.clear();
            is_walls_u.clear();
            is_walls_v.clear();
        }
        bool IsWall(int x, int y, int gridWidth, int gridHeight) const
        {
            // Outer domain boundary
            if (x == 0 || x == gridWidth - 1 || y == 0 || y == gridHeight - 1)
            {
                return true;
            }
            return false;
        }
    };

    struct c_cells_view
    {

        float *divs = nullptr;
        float *smoke = nullptr;
        float *pressures_input = nullptr;
        float *pressures_output = nullptr;
        uint8_t *is_walls = nullptr;
        uint8_t *edges_states_count = nullptr;
        int valid_cell_count = 0;
        int w = -1;
        int h = -1;
    };
    struct c_edges_view
    {
        float *u;
        float *v;
        uint8_t *is_walls_u;
        uint8_t *is_walls_v;
        int edges_w;
        int edges_h;
    };
    namespace CodeSimulationDevice
    {
        __device__ float &GetCellPressure(int x, int y, int w, float *pressures) { return pressures[y * w + x]; }

        __device__ float GetCellFluidState(int x, int y, int w, uint8_t *is_walls)
        {
            return is_walls[y * w + x] ? 0.0f : 1.0f;
        }

        __device__ uint8_t &GetCellEdgesStateCount(int x, int y, int w, uint8_t *edges_states)
        {
            return edges_states[y * w + x];
        }

        __device__ float &GetEdge(int x, int y, int edges_w, float *uv) { return uv[y * edges_w + x]; }


        __device__ void GetCellEdgesIdxs(int x, int y, int edge_w, int &edge_u_left_out, int &edge_u_right_out,
                                         int &edge_v_top_out, int &edge_v_bottom_out)
        {

            edge_u_left_out = y * edge_w + x;
            edge_u_right_out = y * edge_w + (x + 1);

            edge_v_top_out = (y + 1) * edge_w + x;
            edge_v_bottom_out = y * edge_w + x;
        }

        __device__ float GetEdgeState(int x, int y, int edges_w, uint8_t *uv_edges_state_arr)
        {
            return uv_edges_state_arr[y * edges_w + x] == 1 ? 0.0f : 1.0f;
        }

        __device__ uint8_t &GetWall(int x, int y, int edges_w, uint8_t *is_walls_uv)
        {
            return is_walls_uv[y * edges_w + x];
        }


        __global__ void k_simulation_projection(int size, float density, float dx, float dt, c_cells_view cells_data,
                                                c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            if (cells_data.is_walls[idx] == 1)
            {
                return;
            }
            int x = idx % cells_data.w;
            int y = idx / cells_data.w;
            uint8_t s = GetCellEdgesStateCount(x, y, cells_data.w, cells_data.edges_states_count);
            if (s == 0)
            {
                // CODECUDA_PRINTLN("solid");
                return;
                ;
            }

            int edge_u_left_out_idx = -1;
            int edge_u_right_out_idx = -1;
            int edge_v_top_out_idx = -1;
            int edge_v_bottom_out_idx = -1;
            float press_l = GetCellPressure(x - 1, y, cells_data.w, cells_data.pressures_input) *
                GetCellFluidState(x - 1, y, cells_data.w, cells_data.is_walls);
            float press_r = GetCellPressure(x + 1, y, cells_data.w, cells_data.pressures_input) *
                GetCellFluidState(x + 1, y, cells_data.w, cells_data.is_walls);
            float press_t = GetCellPressure(x, y + 1, cells_data.w, cells_data.pressures_input) *
                GetCellFluidState(x, y + 1, cells_data.w, cells_data.is_walls);
            float press_b = GetCellPressure(x, y - 1, cells_data.w, cells_data.pressures_input) *
                GetCellFluidState(x, y - 1, cells_data.w, cells_data.is_walls);

            GetCellEdgesIdxs(x, y, edges_view.edges_w, edge_u_left_out_idx, edge_u_right_out_idx, edge_v_top_out_idx,
                             edge_v_bottom_out_idx);

            float press_sum = (press_l + press_r + press_t + press_b);
            float u_r = GetEdge(x + 1, y, edges_view.edges_w, edges_view.u) *
                GetEdgeState(x + 1, y, edges_view.edges_w, edges_view.is_walls_u);
            float u_l = GetEdge(x, y, edges_view.edges_w, edges_view.u) *
                GetEdgeState(x, y, edges_view.edges_w, edges_view.is_walls_u);
            float v_t = GetEdge(x, y + 1, edges_view.edges_w, edges_view.v) *
                GetEdgeState(x, y + 1, edges_view.edges_w, edges_view.is_walls_v);
            float v_b = GetEdge(x, y, edges_view.edges_w, edges_view.v) *
                GetEdgeState(x, y, edges_view.edges_w, edges_view.is_walls_v);

            float velocities_sum = u_r - u_l + v_t - v_b;
            float pressure_new = (press_sum / float(s)) - (density * dx * velocities_sum) / (float(s) * dt);
            cells_data.pressures_output[idx] = pressure_new;
        }

        __global__ void k_simulation_update_velocities_u(int size, float k, c_cells_view cells_data,
                                                         c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            if (edges_view.is_walls_u[idx] == 0)
            {
                int x = idx % edges_view.edges_w;
                int y = idx / edges_view.edges_w;
                float press_r = GetCellPressure(x, y, cells_data.w, cells_data.pressures_output);
                float press_l = GetCellPressure(x - 1, y, cells_data.w, cells_data.pressures_output);
                edges_view.u[idx] = edges_view.u[idx] - (k * (press_r - press_l));
            }
            else
            {
                edges_view.u[idx] = 0.0f;
            }
        }

        __global__ void k_simulation_update_velocities_v(int size, float dt, float gravity, float k,
                                                         c_cells_view cells_data, c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            if (edges_view.is_walls_v[idx] == 0)
            {
                int x = idx % edges_view.edges_w;
                int y = idx / edges_view.edges_w;

                float press_t = GetCellPressure(x, y, cells_data.w, cells_data.pressures_output);
                float press_b = GetCellPressure(x, y - 1, cells_data.w, cells_data.pressures_output);
                edges_view.v[idx] = edges_view.v[idx] - (k * (press_t - press_b));
                // edges_view.v[idx] += (dt * (gravity));
            }
            else
            {
                edges_view.v[idx] = 0.0f;
            }

        }
    } // namespace CodeSimulationDevice

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

            edges_data.Resize(this->edge_w, this->edge_h);

            cells_data.Resize(w, h);

            // First pass: mark all solid-cell edges.
            for (int i = 0; i < cells_data.pressures.size(); ++i)
            {
                if (!cells_data.is_walls[i])
                {
                    continue;
                }

                int x = i % w;
                int y = i / w;

                int u_left;
                int u_right;
                int v_top;
                int v_bottom;

                GetCellEdgesIdxs(x, y, u_left, u_right, v_top, v_bottom);

                edges_data.is_walls_u[u_left] = 1;
                edges_data.is_walls_u[u_right] = 1;
                edges_data.is_walls_v[v_top] = 1;
                edges_data.is_walls_v[v_bottom] = 1;

                edges_data.u[u_left] = 0.0f;
                edges_data.u[u_right] = 0.0f;
                edges_data.v[v_top] = 0.0f;
                edges_data.v[v_bottom] = 0.0f;
            }

            // Second pass: calculate counts after every wall is known.
            for (int i = 0; i < cells_data.pressures.size(); ++i)
            {
                int x = i % w;
                int y = i / w;

                cells_data.edges_states_count[i] =
                    static_cast<uint8_t>(edges_data.GetStateU(x, y) + edges_data.GetStateU(x + 1, y) +
                                         edges_data.GetStateV(x, y) + edges_data.GetStateV(x, y + 1));
            }
            InitViews();
            ready_to_run = true;
        }
        void InitViews()
        {
            int cell_count = w * h;
            int edge_count = edge_w * edge_h;

            cells_view.w = w;
            cells_view.h = h;
            CODE_API::CW_Malloc(&cells_view.divs, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.smoke, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.pressures_input, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.pressures_output, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.is_walls, sizeof(uint8_t) * cell_count);
            CODE_API::CW_Malloc(&cells_view.edges_states_count, sizeof(uint8_t) * cell_count);


            edges_view.edges_w = edge_w;
            edges_view.edges_h = edge_h;
            CODE_API::CW_Malloc(&edges_view.u, sizeof(float) * edge_count);
            CODE_API::CW_Malloc(&edges_view.v, sizeof(float) * edge_count);
            CODE_API::CW_Malloc(&edges_view.is_walls_u, sizeof(uint8_t) * edge_count);
            CODE_API::CW_Malloc(&edges_view.is_walls_v, sizeof(uint8_t) * edge_count);

            CODE_API::CW_Memcpy(cells_view.divs, cells_data.divs.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.smoke, cells_data.smoke.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.pressures_input, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.pressures_output, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.is_walls, cells_data.is_walls.data(), sizeof(uint8_t) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.edges_states_count, cells_data.edges_states_count.data(),
                                sizeof(uint8_t) * cell_count, cudaMemcpyHostToDevice);

            CODE_API::CW_Memcpy(edges_view.u, edges_data.u.data(), sizeof(float) * edge_count, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.v, edges_data.v.data(), sizeof(float) * edge_count, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_u, edges_data.is_walls_u.data(), sizeof(uint8_t) * edge_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_v, edges_data.is_walls_v.data(), sizeof(uint8_t) * edge_count,
                                cudaMemcpyHostToDevice);
        }
        void CopyHostToDevice(float *current_pressure)
        {

            int cell_count = w * h;
            int edge_count = edge_w * edge_h;
            CODE_API::CW_Memcpy(cells_view.divs, cells_data.divs.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.smoke, cells_data.smoke.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(current_pressure, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.is_walls, cells_data.is_walls.data(), sizeof(uint8_t) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.edges_states_count, cells_data.edges_states_count.data(),
                                sizeof(uint8_t) * cell_count, cudaMemcpyHostToDevice);

            CODE_API::CW_Memcpy(edges_view.u, edges_data.u.data(), sizeof(float) * edge_count, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.v, edges_data.v.data(), sizeof(float) * edge_count, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_u, edges_data.is_walls_u.data(), sizeof(uint8_t) * edge_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_v, edges_data.is_walls_v.data(), sizeof(uint8_t) * edge_count,
                                cudaMemcpyHostToDevice);
        }

        void CopyDeviceToHost(float *current_pressure)
        {
            int cell_count = w * h;
            int edge_count = edge_w * edge_h;

            CODE_API::CW_Memcpy(cells_data.divs.data(), cells_view.divs, sizeof(float) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.smoke.data(), cells_view.smoke, sizeof(float) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.pressures.data(), current_pressure, sizeof(float) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.is_walls.data(), cells_view.is_walls, sizeof(uint8_t) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.edges_states_count.data(), cells_view.edges_states_count,
                                sizeof(uint8_t) * cell_count, cudaMemcpyDeviceToHost);

            CODE_API::CW_Memcpy(edges_data.u.data(), edges_view.u, sizeof(float) * edge_count, cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.v.data(), edges_view.v, sizeof(float) * edge_count, cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.is_walls_u.data(), edges_view.is_walls_u, sizeof(uint8_t) * edge_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.is_walls_v.data(), edges_view.is_walls_v, sizeof(uint8_t) * edge_count,
                                cudaMemcpyDeviceToHost);
        }
        void ClearViews()
        {
            CODE_API::CW_Free(cells_view.divs);
            CODE_API::CW_Free(cells_view.pressures_input);
            CODE_API::CW_Free(cells_view.pressures_output);
            CODE_API::CW_Free(cells_view.is_walls);
            CODE_API::CW_Free(cells_view.edges_states_count);

            CODE_API::CW_Free(edges_view.u);
            CODE_API::CW_Free(edges_view.v);
            CODE_API::CW_Free(edges_view.is_walls_u);
            CODE_API::CW_Free(edges_view.is_walls_v);
        }
        
        void FreeSim()
        {
            cells_data.Reset();
            edges_data.Reset();
            ClearViews();
            ready_to_run = false;
        }
        
        void RestartSim()
        {
            assert(w > 0);
            assert(h > 0);
            FreeSim();
            InitGrid(w, h);
        }

        void ProjectionGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            dim3 grid((double(w * h) + 1023.0) / 1024.0, 1, 1);
            for (int iter = 0; iter < total_iter_gpu; ++iter)
            {
                CodeSimulationDevice::k_simulation_projection<<<grid, block, 0, stream>>>(w * h, density, dx, dt,
                                                                                          cells_view, edges_view);
                if (iter < total_iter_gpu - 1)
                {
                    std::swap(cells_view.pressures_input, cells_view.pressures_output);
                }
            }
        }
        void UpdateSimulationGPU(cudaStream_t stream)
        {
            AdvectVelocity();
            CopyHostToDevice(cells_view.pressures_input);
            ProjectionGPU(stream);
            UpdateVelocityGPU(stream);
            CODE_API::CW_StreamSynchronize(stream);
            CopyDeviceToHost(cells_view.pressures_output);
            if (debug)
            {
                ProjectionResults(total_iter_gpu);
            }
            UpdateData();
        }


        void UpdateSimulationCPU()
        {
            AdvectVelocity();
            Projection();
            ProjectionResults(total_iter_cpu);
            UpdateVelocity();
            UpdateData();
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

                    const int sq_dist = x * x + y * y;

                    if (sq_dist == 0 || sq_dist > radius_sq)
                    {
                        continue;
                    }

                    const int idx = y_final * edge_w + x_final;

                    const float distance = std::sqrt(static_cast<float>(sq_dist));

                    const float u = static_cast<float>(x) / distance * scale;

                    const float v = static_cast<float>(y) / distance * scale;

                    if (!edges_data.is_walls_u[idx])
                    {
                        edges_data.u[idx] += u;
                    }

                    if (!edges_data.is_walls_v[idx])
                    {
                        edges_data.v[idx] += v;
                    }
                }
            }
        }
        
        void AddSmoke(int x_pos, int y_pos, int radius, float value)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h)
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
                    if (x_final < 0 || x_final >= w || y_final < 0 || y_final >= h)
                    {
                        continue;
                    }
                    int sq_dist = pow(x_final - x_pos, 2.0f) + pow(y_final - y_pos, 2.0f);
                    if (sq_dist >= radius * radius)
                    {
                        continue;
                    }
                    int idx = y_final * w + x_final;
                    if (cells_data.is_walls[idx])
                    {
                        continue;
                    }
                    cells_data.smoke[idx] += value;
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
                    if (edges_data.is_walls_u[idx])
                    {
                        continue;
                    }
                    if (edges_data.is_walls_v[idx])
                    {
                        continue;
                    }

                    edges_data.u[idx] += vel_x;
                    edges_data.v[idx] += vel_y;
                }
            }
        }

    private:
        float GetVFromU(int x, int y, float u, std::vector<float> &v_edges_old)
        {
            float tl_v = v_edges_old[(y + 1) * edge_w + x];
            float tr_v = v_edges_old[(y + 1) * edge_w + (x + 1)];
            float bl_v = v_edges_old[y * edge_w + x];
            float br_v = v_edges_old[y * edge_w + (x + 1)];
            float v = (tl_v + tr_v + bl_v + br_v) * 0.25f;
            return v;
        }

        float GetUFromV(int x, int y, float v, std::vector<float> &u_edges_old)
        {
            float tl_u = u_edges_old[(y + 1) * edge_w + x];
            float tr_u = u_edges_old[(y + 1) * edge_w + (x + 1)];
            float bl_u = u_edges_old[y * edge_w + x];
            float br_u = u_edges_old[y * edge_w + (x + 1)];
            float u = (tl_u + tr_u + bl_u + br_u) * 0.25f;
            return u;
        }
        
        float SampleSmoke(float x, float y, int cells_w, int cells_h, std::vector<float> &smoke_cells)
        {

            x = std::clamp(x, 0.0f, float(cells_w - 2));
            y = std::clamp(y, 0.0f, float(cells_h - 2));
            float tl_u_prev = smoke_cells[int(y + 1) * cells_w + int(x)];
            float tr_u_prev = smoke_cells[int(y + 1) * cells_w + (int(x) + 1)];
            float bl_u_prev = smoke_cells[(int(y)) * cells_w + int(x)];
            float br_u_prev = smoke_cells[(int(y)) * cells_w + (int(x) + 1)];

            float wx = x - floor(x);
            float wy = y - floor(y);

            float top = tl_u_prev * (1.0f - wx) + tr_u_prev * (wx);
            float bot = bl_u_prev * (1.0f - wx) + br_u_prev * (wx);

            return top * (wy) + bot * (1.0f - wy);
        }
        float SampleEdge(float x, float y, int edge_w_in, int edge_h_in, std::vector<float> &edges_old)
        {

            x = std::clamp(x, 0.0f, float(edge_w_in - 2));
            y = std::clamp(y, 0.0f, float(edge_h_in - 2));
            float tl_u_prev = edges_old[int(y + 1) * edge_w_in + int(x)];
            float tr_u_prev = edges_old[int(y + 1) * edge_w_in + (int(x) + 1)];
            float bl_u_prev = edges_old[(int(y)) * edge_w_in + int(x)];
            float br_u_prev = edges_old[(int(y)) * edge_w_in + (int(x) + 1)];

            float wx = x - floor(x);
            float wy = y - floor(y);

            float top = tl_u_prev * (1.0f - wx) + tr_u_prev * (wx);
            float bot = bl_u_prev * (1.0f - wx) + br_u_prev * (wx);

            return top * (wy) + bot * (1.0f - wy);
        }
        void AdvectVelocity()
        {
            std::vector<float> u_edges_old = edges_data.u;
            std::vector<float> v_edges_old = edges_data.v;
            std::vector<float> smoke_cells_old = cells_data.smoke;
            for (int y = 0; y < edge_h; ++y)
            {
                for (int x = 0; x < edge_w; ++x)
                {
                    int i = y * edge_w + x;
                    if (edges_data.is_walls_u[i])
                    {
                        continue;
                    }

                    float u = u_edges_old[i];
                    float v = GetVFromU(x, y, u, v_edges_old);
                    float pos[2] = {float(x), float(y)};
                    float x_pos = pos[0] - u * dt / dx;
                    float y_pos = pos[1] - v * dt / dy;
                    edges_data.u[i] = SampleEdge(x_pos, y_pos, edge_w, edge_h, u_edges_old);
                }
            }

            for (int y = 0; y < edge_h; ++y)
            {
                for (int x = 0; x < edge_w; ++x)
                {
                    int i = y * edge_w + x;
                    if (edges_data.is_walls_v[i])
                    {
                        continue;
                    }

                    float v = v_edges_old[i];

                    float u = GetUFromV(x, y, v, u_edges_old);

                    float pos[2] = {float(x), float(y)};
                    float x_pos = pos[0] - u * dt / dx;
                    float y_pos = pos[1] - v * dt / dy;
                    edges_data.v[i] = SampleEdge(x_pos, y_pos, edge_w, edge_h, v_edges_old);
                }
            }
            for (int y = 0; y < cells_data.h; ++y)
            {
                for (int x = 0; x < cells_data.w; ++x)
                {
                    int i = y * cells_data.w + x;
                    if (cells_data.is_walls[i])
                    {
                        continue;
                    }
                    int l = -1;
                    int r = -1;
                    int b = -1;
                    int t = -1;
                    GetCellEdgesIdxs(x, y, l, r, b, t);
                    float u = (edges_data.u[l] + edges_data.u[r]) * 0.5f;
                    float v = (edges_data.v[b] + edges_data.v[t]) * 0.5f;
                    
                    float pos[2] = {float(x), float(y)};
                    float x_pos = pos[0] - u * dt / dx;
                    float y_pos = pos[1] - v * dt / dy; 
                    cells_data.smoke[i] = SampleSmoke(x_pos, y_pos, cells_data.w, cells_data.h, smoke_cells_old);
                }
            }
        }
        void UpdateData()
        {
            sim_step_idx++;
            total_t += dt;
        }
        void UpdateVelocityGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            dim3 grid((double(edge_w * edge_h) + 1023.0) / 1024.0, 1, 1);
            float k = dt / (density * dx);
            CodeSimulationDevice::k_simulation_update_velocities_u<<<grid, block, 0, stream>>>(edge_w * edge_h, k,
                                                                                               cells_view, edges_view);
            k = dt / (density * dy);
            CodeSimulationDevice::k_simulation_update_velocities_v<<<grid, block, 0, stream>>>(
                edge_w * edge_h, dt, g, k, cells_view, edges_view);
        }
        void UpdateVelocity()
        {
            float gravity = (g * gravity_sign);
            float k = dt / (density * dx);
            for (int y = 0; y < edge_h; ++y)
            {
                for (int x = 0; x < edge_w; ++x)
                {
                    if (edges_data.GetWallU(x, y))
                    {
                        edges_data.u[y * edge_w + x] = 0.0f;
                        continue;
                    }

                    float press_r = cells_data.GetCellPressure(x, y);
                    float press_l = cells_data.GetCellPressure(x - 1, y);
                    edges_data.u[y * edge_w + x] = edges_data.u[y * edge_w + x] - (k * (press_r - press_l));
                }
            }
            k = dt / (density * dy);
            for (int y = 0; y < edge_h; ++y)
            {
                for (int x = 0; x < edge_w; ++x)
                {
                    if (edges_data.GetWallV(x, y))
                    {
                        edges_data.v[y * edge_w + x] = 0.0f;
                        continue;
                    }
                    float press_t = cells_data.GetCellPressure(x, y);
                    float press_b = cells_data.GetCellPressure(x, y - 1);
                    edges_data.v[y * edge_w + x] = edges_data.v[y * edge_w + x] - (k * (press_t - press_b));
                }
            }
            // for (int i = 0; i < v_edges.size(); ++i)
            // {
            //     if (v_edges[i].is_wall)
            //     {
            //         v_edges[i].vec = 0.0f;
            //         continue;
            //     }
            //     v_edges[i].vec += (dt * (gravity));
            // }
        }

        void Projection()
        {
            for (int iter = 0; iter < total_iter_cpu; ++iter)
            {
                for (int i = 0; i < cells_data.pressures.size(); ++i)
                {
                    if (cells_data.is_walls[i] == 1)
                    {
                        continue;
                    }
                    int x = i % w;
                    int y = i / w;
                    int s = cells_data.GetCellEdgesStateCount(x, y);
                    if (s == 0)
                    {
                        // CODECUDA_PRINTLN("solid");
                        continue;
                    }

                    int edge_u_left_out_idx = -1;
                    int edge_u_right_out_idx = -1;
                    int edge_v_top_out_idx = -1;
                    int edge_v_bottom_out_idx = -1;
                    float press_l = cells_data.GetCellPressure(x - 1, y) * cells_data.GetCellFluidState(x - 1, y);
                    float press_r = cells_data.GetCellPressure(x + 1, y) * cells_data.GetCellFluidState(x + 1, y);
                    float press_t = cells_data.GetCellPressure(x, y + 1) * cells_data.GetCellFluidState(x, y + 1);
                    float press_b = cells_data.GetCellPressure(x, y - 1) * cells_data.GetCellFluidState(x, y - 1);

                    GetCellEdgesIdxs(x, y, edge_u_left_out_idx, edge_u_right_out_idx, edge_v_top_out_idx,
                                     edge_v_bottom_out_idx);

                    float press_sum = (press_l + press_r + press_t + press_b);
                    float u_r = edges_data.GetU(x + 1, y) * edges_data.GetStateU(x + 1, y);
                    float u_l = edges_data.GetU(x, y) * edges_data.GetStateU(x, y);
                    float v_t = edges_data.GetV(x, y + 1) * edges_data.GetStateV(x, y + 1);
                    float v_b = edges_data.GetV(x, y) * edges_data.GetStateV(x, y);

                    float velocities_sum = u_r - u_l + v_t - v_b;
                    float pressure_old = cells_data.pressures[i];
                    float pressure_new = (press_sum / float(s)) - (density * dx * velocities_sum) / (float(s) * dt);
                    cells_data.pressures[i] = pressure_new;
                }
            }
        }
        void ProjectionResults(int iters)
        {
            int converged = 0;
            for (int i = 0; i < cells_data.pressures.size(); ++i)
            {

                if (cells_data.is_walls[i])
                {
                    continue;
                }
                int x = i % w;
                int y = i / w;
                int s = cells_data.GetCellEdgesStateCount(x, y);
                if (s == 0)
                {
                    // CODECUDA_PRINTLN("solid");
                    continue;
                }
                const float u_r = edges_data.GetU(x + 1, y) * edges_data.GetStateU(x + 1, y);

                const float u_l = edges_data.GetU(x, y) * edges_data.GetStateU(x, y);

                const float v_t = edges_data.GetV(x, y + 1) * edges_data.GetStateV(x, y + 1);

                const float v_b = edges_data.GetV(x, y) * edges_data.GetStateV(x, y);

                cells_data.divs[i] = Overrelaxation(u_r - u_l + v_t - v_b);

                if (std::abs(cells_data.divs[i]) < epsilon)
                {
                    converged++;
                };
            }
            PrintDivergenceConvergence(iters);
        }


        void GetCellEdgesIdxs(int x, int y, int &edge_u_left_out, int &edge_u_right_out, int &edge_v_top_out,
                              int &edge_v_bottom_out)
        {

            edge_u_left_out = y * edge_w + x;
            edge_u_right_out = y * edge_w + (x + 1);

            edge_v_top_out = (y + 1) * edge_w + x;
            edge_v_bottom_out = y * edge_w + x;
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
            float minV = std::numeric_limits<float>::max();
            float maxV = std::numeric_limits<float>::lowest();
            int validUCount = 0;

            float sumAbsV = 0.0f;
            float maxAbsV = 0.0f;
            int validVCount = 0;

            for (int i = 0; i < cells_data.pressures.size(); ++i)
            {
                if (cells_data.is_walls[i])
                {
                    continue;
                }

                const float absDiv = std::abs(cells_data.divs[i]);
                const float absPres = std::abs(cells_data.pressures[i]);

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

            for (int i = 0; i < edges_data.u.size(); ++i)
            {
                if (edges_data.is_walls_u[i])
                {
                    continue;
                }

                const float u = edges_data.u[i];
                const float absU = std::abs(u);

                sumAbsU += absU;
                maxAbsU = std::max(maxAbsU, absU);
                minU = std::min(minU, u);
                maxU = std::max(maxU, u);
                validUCount++;
            }

            for (int i = 0; i < edges_data.u.size(); ++i)
            {
                if (edges_data.is_walls_v[i])
                {
                    continue;
                }

                const float v = edges_data.v[i];
                const float absV = std::abs(v);

                sumAbsV += absV;
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
                maxAbsV = std::max(maxAbsV, absV);
                validVCount++;
            }

            const float avgAbsDiv = totalCells > 0 ? sumAbsDiv / static_cast<float>(totalCells) : 0.0f;

            const float avgAbsPres = totalCells > 0 ? sumAbsPres / static_cast<float>(totalCells) : 0.0f;

            const float avgAbsU = validUCount > 0 ? sumAbsU / static_cast<float>(validUCount) : 0.0f;

            const float avgAbsV = validVCount > 0 ? sumAbsV / static_cast<float>(validVCount) : 0.0f;

            const float avgSpeed = std::sqrt(avgAbsU * avgAbsU + avgAbsV * avgAbsV);


            if (validUCount == 0)
            {
                minU = 0.0f;
                maxU = 0.0f;
            }

            std::cout << std::setprecision(2) << "step=" << sim_step_idx << " | time=" << total_t << "s"
                      << " | iter=" << iteration << " | converged=" << convergedCells << "/" << totalCells
                      << " | div(avg/max)=" << avgAbsDiv << "/" << maxAbsDiv << " | pressure(avg/max)=" << avgAbsPres
                      << "/" << maxAbsPres << " | u(avg/max/range)=" << avgAbsU << "/" << maxAbsU << "/[" << minU << ","
                      << maxU << "]"
                      << " | v(avg/max/range)=" << avgAbsV << "/" << maxAbsV << "/[" << minV << "," << maxV << "]"
                      << '\n';
        }

        bool IsSolidCell(int x, int y) const
        {
            if (x == 0 || x == w - 1 || y == 0 || y == h - 1)
            {
                return true;
            }

            const float px = (float(x) + 0.5f) / float(w);
            const float py = (float(y) + 0.5f) / float(h);

            const float dx = px - 0.5f;
            const float dy = py - 0.5f;

            constexpr float radius = 0.05f;

            return dx * dx + dy * dy <= radius * radius;
        }
        bool IsWall(int x, int y, int gridWidth, int gridHeight) const
        {
            // Outer domain boundary
            if (x == 0 || x == gridWidth - 1 || y == 0 || y == gridHeight - 1)
            {
                return true;
            }
            return false;

            // // Convert the integer coordinate to normalized [0, 1] space.
            // const float normalizedX = float(x) / float(gridWidth - 1);
            // const float normalizedY = float(y) / float(gridHeight - 1);
            //
            // const float centerX = 0.5f;
            // const float centerY = 0.5f;
            //
            // // Radius relative to the domain size.
            // const float radius = 0.15f;
            //
            // const float offsetX = normalizedX - centerX;
            // const float offsetY = normalizedY - centerY;
            //
            // return offsetX * offsetX + offsetY * offsetY <= radius * radius;
        }
        const float epsilon = 0.0001f;
        float dt = 1.0f / 60.0f;
        float g = 9.8f;
        float dx = 0.0f;
        float dy = 0.0f;

    public:
        bool ready_to_run = false;
        int w = -1;
        int h = -1;
        int edge_w = -1;
        int edge_h = -1;
        c_cells_view cells_view = {};
        c_edges_view edges_view = {};
        c_cells cells_data = {};
        c_edges edges_data = {};
        int gravity_sign = 1;
        float density = 1.0;
        int64_t sim_step_idx = 0;
        float total_t = 0.0f;
        float weight_sor = 1.6f;
        int total_iter_gpu = 8500;
        int total_iter_cpu = 12;
        bool debug = false;
    };


} // namespace CodeSimulation
#endif // CODECOMMON_HPP
