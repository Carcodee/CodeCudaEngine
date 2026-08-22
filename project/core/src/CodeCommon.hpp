//
// Created by carlo on 2026-02-01.
//

#ifndef CODECOMMON_HPP
#define CODECOMMON_HPP


#include <algorithm>

#include "CodeSimulationParams.hpp"
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
namespace CodeCuda::FluidSimulation
{

    inline constexpr float epsilon = 1e-4f;
    using namespace code_math;
    using sim_params = sim_params;
    struct c_cells
    {
        std::vector<vec4> smoke;
        std::vector<vec2> solid_speeds;
        std::vector<float> divs;
        std::vector<float> pressures;
        std::vector<uint8_t> is_walls;
        std::vector<uint8_t> edges_states_count;
        int w = -1;
        int h = -1;

        void Resize(int w, int h)
        {
            this->w = w;
            this->h = h;
            divs.resize(w * h);
            pressures.resize(w * h);
            is_walls.resize(w * h, 0);
            solid_speeds.resize(w * h);
            smoke.resize(w * h);
            edges_states_count.resize(w * h);

            // for (int i = 0; i < smoke.size(); ++i)
            // {
            //     float rx = float(rand() % 1000) / 1000.0f;
            //     float ry = float(rand() % 1000) / 1000.0f;
            //     float rz = float(rand() % 1000) / 1000.0f;
            //     smoke[i] = vec4(rx, ry, rz);
            //
            // }
            for (int i = 0; i < is_walls.size(); ++i)
            {
                int x = i % w;
                int y = i / w;
                // is_walls[i] = IsBoundary(x, y);
                solid_speeds[i] = vec2(0.0, 0.0);
            }
        }
        void Reset()
        {
            this->w = w;
            this->h = h;
            divs.clear();
            pressures.clear();
            solid_speeds.clear();
            is_walls.clear();
            smoke.clear();
            edges_states_count.clear();
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

        [[nodiscard]]
        uint8_t IsBoundary(int x, int y) const
        {
            if (x == 0 || x == w - 1 || y == 0 || y == h - 1)
            {
                return 1;
            }
            return 0;
        }
    };
    struct c_edges
    {
        int edges_w_u = -1;
        int edges_h_u = -1;
        int edges_w_v = -1;
        int edges_h_v = -1;
        std::vector<float> u;
        std::vector<float> v;
        std::vector<uint8_t> is_walls_u;
        std::vector<uint8_t> is_walls_v;

        float &GetV(int x, int y) { return v[y * edges_w_v + x]; }
        float &GetU(int x, int y) { return u[y * edges_w_u + x]; }

        float GetStateU(int x, int y) { return is_walls_u[y * edges_w_u + x] == 1 ? 0.0f : 1.0f; }
        float GetStateV(int x, int y) { return is_walls_v[y * edges_w_v + x] == 1 ? 0.0f : 1.0f; }

        uint8_t &GetWallU(int x, int y) { return is_walls_u[y * edges_w_u + x]; }

        uint8_t &GetWallV(int x, int y) { return is_walls_v[y * edges_w_v + x]; }
        void Resize(int w_u, int h_u, int w_v, int h_v)
        {
            edges_w_u = w_u;
            edges_h_u = h_u;
            edges_w_v = w_v;
            edges_h_v = h_v;
            u.resize(w_u * h_u);
            v.resize(w_v * h_v);
            is_walls_u.resize(w_u * h_u);
            is_walls_v.resize(w_v * h_v);

            for (int i = 0; i < u.size(); ++i)
            {
                int x = i % edges_w_u;
                is_walls_u[i] = IsWallU(x, edges_w_u);
                u[i] = 0.001f;
            }
            for (int i = 0; i < v.size(); ++i)
            {
                int y = i / edges_w_v;
                is_walls_v[i] = IsWallV(y, edges_h_v);
                v[i] = 0.001f;
            }
        }

        void Reset()
        {
            edges_w_u = -1;
            edges_h_u = -1;
            edges_w_v = -1;
            edges_h_v = -1;
            u.clear();
            v.clear();
            is_walls_u.clear();
            is_walls_v.clear();
        }
        [[nodiscard]]
        bool IsWallU(int x, int grid_width) const
        {
            return x == 0 || x == grid_width - 1;
        }

        [[nodiscard]]
        bool IsWallV(int y, int grid_height) const
        {
            return y == 0 || y == grid_height - 1;
        }

        float GetActiveU(int x, int y) const
        {
            if (x < 0 || x >= edges_w_u || y < 0 || y >= edges_h_u)
            {
                return 0.0f;
            }
            const int idx = y * edges_w_u + x;
            return is_walls_u[idx] ? 0.0f : u[idx];
        }

        float GetActiveV(int x, int y) const
        {
            if (x < 0 || x >= edges_w_v || y < 0 || y >= edges_h_v)
            {
                return 0.0f;
            }
            const int idx = y * edges_w_v + x;
            return is_walls_v[idx] ? 0.0f : v[idx];
        }
    };

    struct c_cells_view
    {

        vec4 *smoke_input = nullptr;
        vec4 *smoke_output = nullptr;
        vec2 *solid_speeds = nullptr;
        float *divs = nullptr;
        float *pressures_input = nullptr;
        float *pressures_output = nullptr;
        uint8_t *is_walls = nullptr;
        uint8_t *edges_states_count = nullptr;
        int w = -1;
        int h = -1;
    };
    struct c_edges_view
    {

        float *u_input;
        float *v_input;
        float *u_output;
        float *v_output;
        uint8_t *is_walls_u;
        uint8_t *is_walls_v;
        int edges_w_u;
        int edges_h_u;
        int edges_w_v;
        int edges_h_v;
    };
    namespace CodeSimulationDevice
    {
        __device__ float &GetCellPressure(int x, int y, int w, int h, float *pressures)
        {
            if (x < 0 || y < 0 || x >= w || y >= h)
            {
                return pressures[0];
            }
            return pressures[y * w + x];
        }

        __device__ vec4 &GetCellSmoke(int x, int y, int w, int h, vec4 *values)
        {
            if (x < 0 || y < 0 || x >= w || y >= h)
            {
                return values[0];
            }
            return values[y * w + x];
        }

        __device__ float GetCellFluidState(int x, int y, int w, int h, uint8_t *is_walls, vec2 *solid_vel = nullptr)
        {
            if (x < 0 || y < 0 || x >= w || y >= h)
            {
                return 0.0f;
            }
            // if (solid_vel != nullptr)
            // {
            //     return is_walls[y * w + x] == 1 && solid_vel[y * w + x].x < epsilon && solid_vel[y * w + x].y <
            //     epsilon
            //         ? 0.0f
            //         : 1.0f;
            // }
            // else
            // {
            //     return is_walls[y * w + x] == 1 ? 0.0f : 1.0f;
            // }
            return is_walls[y * w + x] == 1 ? 0.0f : 1.0f;
        }

        __device__ uint8_t &GetCellEdgesStateCount(int x, int y, int w, uint8_t *edges_states)
        {
            return edges_states[y * w + x];
        }

        __device__ float &GetEdge(int x, int y, int edges_w, float *uv) { return uv[y * edges_w + x]; }


        __device__ void GetCellEdgesIdxs(int x, int y, int edges_w_u, int edges_w_v, int &edge_u_left_out,
                                         int &edge_u_right_out, int &edge_v_top_out, int &edge_v_bottom_out)
        {

            edge_u_left_out = y * edges_w_u + x;
            edge_u_right_out = y * edges_w_u + (x + 1);

            edge_v_top_out = (y + 1) * edges_w_v + x;
            edge_v_bottom_out = y * edges_w_v + x;
        }

        __device__ float GetEdgeState(int x, int y, int edges_w, uint8_t *uv_edges_state_arr)
        {
            return uv_edges_state_arr[y * edges_w + x] == 1 ? 0.0f : 1.0f;
        }

        __device__ float GetActiveEdge(int x, int y, int edges_w, int edges_h, float *uv,
                                       uint8_t *uv_edges_state_arr)
        {
            if (x < 0 || x >= edges_w || y < 0 || y >= edges_h)
            {
                return 0.0f;
            }
            const int idx = y * edges_w + x;
            return uv_edges_state_arr[idx] == 1 ? 0.0f : uv[idx];
        }

        __device__ uint8_t &GetWall(int x, int y, int edges_w, uint8_t *is_walls_uv)
        {
            return is_walls_uv[y * edges_w + x];
        }


        __device__ inline float clamp(float val, float lo, float hi) { return fminf(fmaxf(val, lo), hi); }

        // For double-precision floating point (double)
        __device__ inline double clamp(double val, double lo, double hi) { return fmin(fmax(val, lo), hi); }

        // For integers
        __device__ inline int clamp(int val, int lo, int hi) { return val < lo ? lo : (val > hi ? hi : val); }
        __device__ float SampleEdge(float x, float y, int edge_w_in, int edge_h_in, float *edges_old)
        {

            x = clamp(x, 1.0f, float(edge_w_in - 2));
            y = clamp(y, 1.0f, float(edge_h_in - 2));
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

        template <typename T>
        __device__ T SampleQuantity(float x, float y, int cells_w, int cells_h, const T *cell_quantity)
        {

            x = clamp(x, 1.0f, float(cells_w - 2));
            y = clamp(y, 1.0f, float(cells_h - 2));
            const T tl_u_prev = cell_quantity[int(y + 1) * cells_w + int(x)];
            const T tr_u_prev = cell_quantity[int(y + 1) * cells_w + (int(x) + 1)];
            const T bl_u_prev = cell_quantity[int(y) * cells_w + int(x)];
            const T br_u_prev = cell_quantity[int(y) * cells_w + (int(x) + 1)];

            float wx = x - floor(x);
            float wy = y - floor(y);

            const T top = tl_u_prev * (1.0f - wx) + tr_u_prev * wx;
            const T bottom = bl_u_prev * (1.0f - wx) + br_u_prev * wx;

            return top * wy + bottom * (1.0f - wy);
        }
        __global__ void k_apply_forces(int size_u, int size_v, float wind, float g, float diffusion_factor, float dt,
                                       c_cells_view cells_data, c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            float acc = wind;
            if (idx < size_u && edges_view.is_walls_u[idx] == 0)
            {
                edges_view.u_output[idx] *= (1.0f - diffusion_factor);
                edges_view.u_output[idx] += acc * dt;
            }
            if (idx < size_v && edges_view.is_walls_v[idx] == 0)
            {
                edges_view.v_output[idx] *= (1.0f - diffusion_factor);
                edges_view.v_output[idx] += g * dt;
            }
        }


        __global__ void k_diffuse(int size_u, int size_v, float viscosity, float dt, c_cells_view cells_data,
                                  c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            float v = viscosity;
            float a = v * dt;
            float denom = 1 + 4 * a;
            if (idx < size_u && edges_view.is_walls_u[idx] == 0)
            {
                int x = idx % edges_view.edges_w_u;
                int y = idx / edges_view.edges_w_u;
                float l = GetActiveEdge(x - 1, y, edges_view.edges_w_u, edges_view.edges_h_u, edges_view.u_output,
                                        edges_view.is_walls_u);
                float r = GetActiveEdge(x + 1, y, edges_view.edges_w_u, edges_view.edges_h_u, edges_view.u_output,
                                        edges_view.is_walls_u);
                float b = GetActiveEdge(x, y - 1, edges_view.edges_w_u, edges_view.edges_h_u, edges_view.u_output,
                                        edges_view.is_walls_u);
                float t = GetActiveEdge(x, y + 1, edges_view.edges_w_u, edges_view.edges_h_u, edges_view.u_output,
                                        edges_view.is_walls_u);

                float neightbours_sum = l + r + t + b;
                float u = (neightbours_sum)*a + edges_view.u_output[idx];
                edges_view.u_input[idx] = u / denom;
            }
            if (idx < size_v && edges_view.is_walls_v[idx] == 0)
            {
                int x = idx % edges_view.edges_w_v;
                int y = idx / edges_view.edges_w_v;
                float l = GetActiveEdge(x - 1, y, edges_view.edges_w_v, edges_view.edges_h_v, edges_view.v_output,
                                        edges_view.is_walls_v);
                float r = GetActiveEdge(x + 1, y, edges_view.edges_w_v, edges_view.edges_h_v, edges_view.v_output,
                                        edges_view.is_walls_v);
                float b = GetActiveEdge(x, y - 1, edges_view.edges_w_v, edges_view.edges_h_v, edges_view.v_output,
                                        edges_view.is_walls_v);
                float t = GetActiveEdge(x, y + 1, edges_view.edges_w_v, edges_view.edges_h_v, edges_view.v_output,
                                        edges_view.is_walls_v);

                float neightbours_sum = l + r + t + b;
                float u = (neightbours_sum)*a + edges_view.v_output[idx];
                edges_view.v_input[idx] = u / denom;
            }
        }
        __global__ void k_diffuse_smoke(int size, float smoke_diffuse_coef, float dt, c_cells_view cells_data,
                                        c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % cells_data.w;
            int y = idx / cells_data.w;

            uint8_t s = GetCellEdgesStateCount(x, y, cells_data.w, cells_data.edges_states_count);
            if (s == 0)
            {
                return;
            }

            float d = smoke_diffuse_coef;
            if (GetCellSmoke(x - 1, y, cells_data.w, cells_data.h, cells_data.smoke_output).w > 0.5)
            {
                d = 0.001f;
            }
            float a = d * dt;
            float denom = 1 + s * a;
            if (cells_data.is_walls[idx] == 0)
            {
                vec4 l = GetCellSmoke(x - 1, y, cells_data.w, cells_data.h, cells_data.smoke_output) *
                    GetCellFluidState(x - 1, y, cells_data.w, cells_data.h, cells_data.is_walls);
                vec4 r = GetCellSmoke(x + 1, y, cells_data.w, cells_data.h, cells_data.smoke_output) *
                    GetCellFluidState(x + 1, y, cells_data.w, cells_data.h, cells_data.is_walls);
                vec4 b = GetCellSmoke(x, y - 1, cells_data.w, cells_data.h, cells_data.smoke_output) *
                    GetCellFluidState(x, y - 1, cells_data.w, cells_data.h, cells_data.is_walls);
                vec4 t = GetCellSmoke(x, y + 1, cells_data.w, cells_data.h, cells_data.smoke_output) *
                    GetCellFluidState(x, y + 1, cells_data.w, cells_data.h, cells_data.is_walls);

                vec4 neightbours_sum = l + r + t + b;
                vec4 u = (neightbours_sum)*a + cells_data.smoke_output[idx];
                cells_data.smoke_input[idx] = u / denom;
            }
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
                return;
            }

            int edge_u_left_out_idx = -1;
            int edge_u_right_out_idx = -1;
            int edge_v_top_out_idx = -1;
            int edge_v_bottom_out_idx = -1;
            float press_l = GetCellPressure(x - 1, y, cells_data.w, cells_data.h, cells_data.pressures_input) *
                GetCellFluidState(x - 1, y, cells_data.w, cells_data.h, cells_data.is_walls);
            float press_r = GetCellPressure(x + 1, y, cells_data.w, cells_data.h, cells_data.pressures_input) *
                GetCellFluidState(x + 1, y, cells_data.w, cells_data.h, cells_data.is_walls);
            float press_t = GetCellPressure(x, y + 1, cells_data.w, cells_data.h, cells_data.pressures_input) *
                GetCellFluidState(x, y + 1, cells_data.w, cells_data.h, cells_data.is_walls);
            float press_b = GetCellPressure(x, y - 1, cells_data.w, cells_data.h, cells_data.pressures_input) *
                GetCellFluidState(x, y - 1, cells_data.w, cells_data.h, cells_data.is_walls);

            GetCellEdgesIdxs(x, y, edges_view.edges_w_u, edges_view.edges_w_v, edge_u_left_out_idx,
                             edge_u_right_out_idx, edge_v_top_out_idx, edge_v_bottom_out_idx);

            float press_sum = (press_l + press_r + press_t + press_b);
            float u_r = GetEdge(x + 1, y, edges_view.edges_w_u, edges_view.u_output) *
                GetEdgeState(x + 1, y, edges_view.edges_w_u, edges_view.is_walls_u);
            float u_l = GetEdge(x, y, edges_view.edges_w_u, edges_view.u_output) *
                GetEdgeState(x, y, edges_view.edges_w_u, edges_view.is_walls_u);
            float v_t = GetEdge(x, y + 1, edges_view.edges_w_v, edges_view.v_output) *
                GetEdgeState(x, y + 1, edges_view.edges_w_v, edges_view.is_walls_v);
            float v_b = GetEdge(x, y, edges_view.edges_w_v, edges_view.v_output) *
                GetEdgeState(x, y, edges_view.edges_w_v, edges_view.is_walls_v);

            float velocities_sum = u_r - u_l + v_t - v_b;
            float pressure_new = (press_sum / float(s)) - (density * dx * velocities_sum) / (float(s) * dt);
            cells_data.pressures_output[idx] = pressure_new;
        }

        __global__ void k_simulation_update_velocities_u(int size, float dt, float k, c_cells_view cells_data,
                                                         c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % edges_view.edges_w_u;
            int y = idx / edges_view.edges_w_u;
            // if (x == 0 || x == edges_view.edges_w_u - 1)
            // {
            //     return;
            // }
            if (edges_view.is_walls_u[idx] == 0)
            {
                float press_r = GetCellPressure(x, y, cells_data.w, cells_data.h, cells_data.pressures_input) *
                    GetCellFluidState(x, y, cells_data.w, cells_data.h, cells_data.is_walls);
                float press_l = GetCellPressure(x - 1, y, cells_data.w, cells_data.h, cells_data.pressures_input) *
                    GetCellFluidState(x - 1, y, cells_data.w, cells_data.h, cells_data.is_walls);
                edges_view.u_output[idx] = edges_view.u_output[idx] - (k * (press_r - press_l));
            }
            else
            {
                edges_view.u_output[idx] = 0.0f;
            }
        }

        __global__ void k_simulation_update_velocities_v(int size, float dt, float gravity, float k,
                                                         c_cells_view cells_data, c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % edges_view.edges_w_v;
            int y = idx / edges_view.edges_w_v;

            // if (y == 0 || y == edges_view.edges_h_v - 1)
            // {
            //     return;
            // }
            if (edges_view.is_walls_v[idx] == 0)
            {
                float press_t = GetCellPressure(x, y, cells_data.w, cells_data.h, cells_data.pressures_input) *
                    GetCellFluidState(x, y, cells_data.w, cells_data.h, cells_data.is_walls);
                float press_b = GetCellPressure(x, y - 1, cells_data.w, cells_data.h, cells_data.pressures_input) *
                    GetCellFluidState(x, y - 1, cells_data.w, cells_data.h, cells_data.is_walls);
                edges_view.v_output[idx] = edges_view.v_output[idx] - (k * (press_t - press_b));
            }
            else
            {
                edges_view.v_output[idx] = 0.0f;
            }
        }

        __global__ void k_simulation_advection_u(int size, float dt, float dx, float dy, c_cells_view cells_data,
                                                 c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % edges_view.edges_w_u;
            int y = idx / edges_view.edges_w_u;

            if (!edges_view.is_walls_u[idx])
            {
                float u = edges_view.u_input[idx];
                float v = SampleEdge(float(x) - 0.5f, float(y) + 0.5f, edges_view.edges_w_v,
                                     edges_view.edges_h_v, edges_view.v_input);
                float pos[2] = {float(x), float(y)};
                float x_pos = pos[0] - u * dt / dx;
                float y_pos = pos[1] - v * dt / dy;
                edges_view.u_output[idx] =
                    SampleEdge(x_pos, y_pos, edges_view.edges_w_u, edges_view.edges_h_u, edges_view.u_input);
            }
        }
        __global__ void k_simulation_advection_v(int size, float dt, float dx, float dy, c_cells_view cells_data,
                                                 c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % edges_view.edges_w_v;
            int y = idx / edges_view.edges_w_v;
            if (!edges_view.is_walls_v[idx])
            {
                float v = edges_view.v_input[idx];
                float u = SampleEdge(float(x) + 0.5f, float(y) - 0.5f, edges_view.edges_w_u,
                                     edges_view.edges_h_u, edges_view.u_input);
                float pos[2] = {float(x), float(y)};
                float x_pos = pos[0] - u * dt / dx;
                float y_pos = pos[1] - v * dt / dy;
                edges_view.v_output[idx] =
                    SampleEdge(x_pos, y_pos, edges_view.edges_w_v, edges_view.edges_h_v, edges_view.v_input);
            }
        }
        __global__ void k_simulation_advection_smoke(int size, float dt, float dx, float dy,
                                                     float smoke_disipation_factor, c_cells_view cells_data,
                                                     c_edges_view edges_view)
        {
            uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            int x = idx % cells_data.w;
            int y = idx / cells_data.w;
            if (cells_data.is_walls[idx])
            {
                cells_data.smoke_output[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
                return;
            }
            if (x == cells_data.w - 2)
            {
                cells_data.smoke_output[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
                return;
            }
            int l = -1;
            int r = -1;
            int b = -1;
            int t = -1;
            GetCellEdgesIdxs(x, y, edges_view.edges_w_u, edges_view.edges_w_v, l, r, t, b);
            float u = (edges_view.u_output[l] + edges_view.u_output[r]) * 0.5f;
            float v = (edges_view.v_output[b] + edges_view.v_output[t]) * 0.5f;

            float pos[2] = {float(x), float(y)};
            float x_pos = pos[0] - u * dt / dx;
            float y_pos = pos[1] - v * dt / dy;
            vec4 smoke_q = SampleQuantity(x_pos, y_pos, cells_data.w, cells_data.h, cells_data.smoke_output);
            cells_data.smoke_input[idx] = smoke_q;
            cells_data.smoke_input[idx] = cells_data.smoke_input[idx] * (1.0f - smoke_disipation_factor);
        }
        __global__ void k_simulation_add_velocity(int size, int x_pos, int y_pos, int radius, float vel_x, float vel_y,
                                                  c_edges_view edges_view)
        {
            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
            {
                return;
            }

            const int diameter = radius * 2;

            const int local_x = int(idx % diameter) - radius;
            const int local_y = int(idx / diameter) - radius;

            const int x = x_pos + local_x;
            const int y = y_pos + local_y;

            const int sq_dist = local_x * local_x + local_y * local_y;
            const int radius_sq = radius * radius;

            if (sq_dist >= radius_sq)
            {
                return;
            }

            if (x >= 0 && x < edges_view.edges_w_u && y >= 0 && y < edges_view.edges_h_u)
            {
                const int u_idx = y * edges_view.edges_w_u + x;
                if (!edges_view.is_walls_u[u_idx])
                {
                    edges_view.u_output[u_idx] += vel_x;
                }
            }

            if (x >= 0 && x < edges_view.edges_w_v && y >= 0 && y < edges_view.edges_h_v)
            {
                const int v_idx = y * edges_view.edges_w_v + x;
                if (!edges_view.is_walls_v[v_idx])
                {
                    edges_view.v_output[v_idx] += vel_y;
                }
            }
        }

        __global__ void k_simulation_add_smoke(int size, int x_pos, int y_pos, int radius, vec4 value,
                                               c_cells_view cells_view)
        {
            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
            {
                return;
            }

            const int diameter = radius * 2;

            const int local_x = int(idx % diameter) - radius;
            const int local_y = int(idx / diameter) - radius;

            const int x = x_pos + local_x;
            const int y = y_pos + local_y;

            if (x < 0 || x >= cells_view.w || y < 0 || y >= cells_view.h)
            {
                return;
            }

            const int sq_dist = local_x * local_x + local_y * local_y;
            const int radius_sq = radius * radius;

            if (sq_dist >= radius_sq)
            {
                return;
            }

            const int cell_idx = y * cells_view.w + x;

            if (cells_view.is_walls[cell_idx])
            {
                return;
            }

            cells_view.smoke_output[cell_idx] += value;
        }

        __global__ void k_simulation_add_pressure(int size, int x_pos, int y_pos, int radius, float value,
                                                  c_cells_view cells_view)
        {
            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
            {
                return;
            }

            const int diameter = radius * 2;

            const int local_x = int(idx % diameter) - radius;
            const int local_y = int(idx / diameter) - radius;

            const int x = x_pos + local_x;
            const int y = y_pos + local_y;

            if (x < 0 || x >= cells_view.w || y < 0 || y >= cells_view.h)
            {
                return;
            }

            const int sq_dist = local_x * local_x + local_y * local_y;
            const int radius_sq = radius * radius;

            if (sq_dist >= radius_sq)
            {
                return;
            }

            const int cell_idx = y * cells_view.w + x;

            if (cells_view.is_walls[cell_idx])
            {
                return;
            }

            cells_view.pressures_input[cell_idx] += value;
        }

        __global__ void k_simulation_update_speeds_based_on_solids(int size, c_cells_view cells_view,
                                                                   c_edges_view edges_view)
        {
            uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= size)
            {
                return;
            }
            uint32_t x = idx % cells_view.w;
            uint32_t y = idx / cells_view.w;
            int l = -1;
            int r = -1;
            int t = -1;
            int b = -1;
            GetCellEdgesIdxs(x, y, edges_view.edges_w_u, edges_view.edges_w_v, l, r, t, b);
            vec2 &speed = cells_view.solid_speeds[idx];
            edges_view.u_output[l] = speed.x;
            edges_view.u_output[r] = speed.x;
            edges_view.v_output[b] = speed.y;
            edges_view.v_output[t] = speed.y;
        }
    } // namespace CodeSimulationDevice

    struct c_grid
    {
        c_grid() = default;
        void InitGrid(int width, int height)
        {
            this->edges_w_u = width + 1;
            this->edges_h_u = height;
            this->edges_w_v = width;
            this->edges_h_v = height + 1;
            this->w = width;
            this->h = height;
            this->dx = 1.0f / float(w);
            this->dy = 1.0f / float(h);

            cells_data.Resize(w, h);
            edges_data.Resize(edges_w_u, edges_h_u, edges_w_v, edges_h_v);
            InitSolids();
            InitViews();
            ready_to_run = true;
        }
        void InitSolids()
        {
            // First pass: mark all solid-cell edges.
            for (int i = 0; i < cells_data.w * cells_data.h; ++i)
            {
                if (cells_data.is_walls[i] == 0)
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
                // if (x == 0)
                // {
                //     edges_data.u[u_left] = 0.0f;
                //     continue;
                // }
                // if (x == w - 1)
                // {
                //     edges_data.u[u_right] = 0.0f;
                //     continue;
                // }
                // if (y == 0)
                // {
                //     edges_data.v[v_bottom] = 0.0f;
                //     continue;
                // }
                // if (y == h - 1)
                // {
                //     edges_data.v[v_top] = 0.0f;
                //     continue;
                // }

                edges_data.is_walls_u[u_left] = 1;
                edges_data.is_walls_u[u_right] = 1;
                edges_data.is_walls_v[v_top] = 1;
                edges_data.is_walls_v[v_bottom] = 1;

                edges_data.u[u_left] = cells_data.solid_speeds[i].x;
                edges_data.u[u_right] = cells_data.solid_speeds[i].x;
                edges_data.v[v_top] = cells_data.solid_speeds[i].y;
                edges_data.v[v_bottom] = cells_data.solid_speeds[i].y;
            }

            UpdateCellStates();
        }
        void UpdateCellStates()
        {
            for (int i = 0; i < cells_data.pressures.size(); ++i)
            {
                int x = i % w;
                int y = i / w;

                cells_data.edges_states_count[i] =
                    static_cast<uint8_t>(edges_data.GetStateU(x, y) + edges_data.GetStateU(x + 1, y) +
                                         edges_data.GetStateV(x, y) + edges_data.GetStateV(x, y + 1));
            }
        }
        void InitViews()
        {
            int cell_count = w * h;
            int edge_count_u = edges_w_u * edges_h_u;
            int edge_count_v = edges_w_v * edges_h_v;

            cells_view.w = w;
            cells_view.h = h;
            CODE_API::CW_Malloc(&cells_view.divs, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.solid_speeds, sizeof(vec2) * cell_count);
            CODE_API::CW_Malloc(&cells_view.smoke_input, sizeof(vec4) * cell_count);
            CODE_API::CW_Malloc(&cells_view.smoke_output, sizeof(vec4) * cell_count);
            CODE_API::CW_Malloc(&cells_view.pressures_input, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.pressures_output, sizeof(float) * cell_count);
            CODE_API::CW_Malloc(&cells_view.is_walls, sizeof(uint8_t) * cell_count);
            CODE_API::CW_Malloc(&cells_view.edges_states_count, sizeof(uint8_t) * cell_count);


            edges_view.edges_w_u = edges_w_u;
            edges_view.edges_h_u = edges_h_u;
            edges_view.edges_w_v = edges_w_v;
            edges_view.edges_h_v = edges_h_v;
            CODE_API::CW_Malloc(&edges_view.u_input, sizeof(float) * edge_count_u);
            CODE_API::CW_Malloc(&edges_view.v_input, sizeof(float) * edge_count_v);
            CODE_API::CW_Malloc(&edges_view.u_output, sizeof(float) * edge_count_u);
            CODE_API::CW_Malloc(&edges_view.v_output, sizeof(float) * edge_count_v);
            CODE_API::CW_Malloc(&edges_view.is_walls_u, sizeof(uint8_t) * edge_count_u);
            CODE_API::CW_Malloc(&edges_view.is_walls_v, sizeof(uint8_t) * edge_count_v);

            CODE_API::CW_Memcpy(cells_view.divs, cells_data.divs.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.solid_speeds, cells_data.solid_speeds.data(), sizeof(vec2) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.smoke_input, cells_data.smoke.data(), sizeof(vec4) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.smoke_output, cells_data.smoke.data(), sizeof(vec4) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.pressures_input, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.pressures_output, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.is_walls, cells_data.is_walls.data(), sizeof(uint8_t) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.edges_states_count, cells_data.edges_states_count.data(),
                                sizeof(uint8_t) * cell_count, cudaMemcpyHostToDevice);

            CODE_API::CW_Memcpy(edges_view.u_input, edges_data.u.data(), sizeof(float) * edge_count_u,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.v_input, edges_data.v.data(), sizeof(float) * edge_count_v,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.u_output, edges_data.u.data(), sizeof(float) * edge_count_u,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.v_output, edges_data.v.data(), sizeof(float) * edge_count_v,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_u, edges_data.is_walls_u.data(), sizeof(uint8_t) * edge_count_u,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_v, edges_data.is_walls_v.data(), sizeof(uint8_t) * edge_count_v,
                                cudaMemcpyHostToDevice);
        }
        void CopyHostToDevice(float *current_pressure, vec4 *smoke, float *u, float *v)
        {

            int cell_count = w * h;
            int edge_count_u = edges_w_u * edges_h_u;
            int edge_count_v = edges_w_v * edges_h_v;
            CODE_API::CW_Memcpy(cells_view.divs, cells_data.divs.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.solid_speeds, cells_data.solid_speeds.data(), sizeof(vec2) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.is_walls, cells_data.is_walls.data(), sizeof(uint8_t) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(cells_view.edges_states_count, cells_data.edges_states_count.data(),
                                sizeof(uint8_t) * cell_count, cudaMemcpyHostToDevice);

            CODE_API::CW_Memcpy(edges_view.is_walls_u, edges_data.is_walls_u.data(), sizeof(uint8_t) * edge_count_u,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(edges_view.is_walls_v, edges_data.is_walls_v.data(), sizeof(uint8_t) * edge_count_v,
                                cudaMemcpyHostToDevice);

            CODE_API::CW_Memcpy(current_pressure, cells_data.pressures.data(), sizeof(float) * cell_count,
                                cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(smoke, cells_data.smoke.data(), sizeof(vec4) * cell_count, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(u, edges_data.u.data(), sizeof(float) * edge_count_u, cudaMemcpyHostToDevice);
            CODE_API::CW_Memcpy(v, edges_data.v.data(), sizeof(float) * edge_count_v, cudaMemcpyHostToDevice);
        }

        void CopyDeviceToHost(float *current_pressure, vec4 *smoke, float *u, float *v)
        {
            int cell_count = w * h;
            int edge_count_u = edges_w_u * edges_h_u;
            int edge_count_v = edges_w_v * edges_h_v;

            CODE_API::CW_Memcpy(cells_data.divs.data(), cells_view.divs, sizeof(float) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.solid_speeds.data(), cells_view.solid_speeds, sizeof(vec2) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.is_walls.data(), cells_view.is_walls, sizeof(uint8_t) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.edges_states_count.data(), cells_view.edges_states_count,
                                sizeof(uint8_t) * cell_count, cudaMemcpyDeviceToHost);

            CODE_API::CW_Memcpy(edges_data.is_walls_u.data(), edges_view.is_walls_u, sizeof(uint8_t) * edge_count_u,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.is_walls_v.data(), edges_view.is_walls_v, sizeof(uint8_t) * edge_count_v,
                                cudaMemcpyDeviceToHost);

            CODE_API::CW_Memcpy(cells_data.pressures.data(), current_pressure, sizeof(float) * cell_count,
                                cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(cells_data.smoke.data(), smoke, sizeof(vec4) * cell_count, cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.u.data(), u, sizeof(float) * edge_count_u, cudaMemcpyDeviceToHost);
            CODE_API::CW_Memcpy(edges_data.v.data(), v, sizeof(float) * edge_count_v, cudaMemcpyDeviceToHost);
        }
        void ClearViews()
        {
            CODE_API::CW_Free(cells_view.divs);
            CODE_API::CW_Free(cells_view.solid_speeds);
            CODE_API::CW_Free(cells_view.pressures_input);
            CODE_API::CW_Free(cells_view.pressures_output);
            CODE_API::CW_Free(cells_view.smoke_input);
            CODE_API::CW_Free(cells_view.smoke_output);
            CODE_API::CW_Free(cells_view.is_walls);
            CODE_API::CW_Free(cells_view.edges_states_count);

            CODE_API::CW_Free(edges_view.u_input);
            CODE_API::CW_Free(edges_view.v_input);
            CODE_API::CW_Free(edges_view.u_output);
            CODE_API::CW_Free(edges_view.v_output);
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

        void UpdateSimulation(cudaStream_t stream = nullptr)
        {

            if (params.gpu_sim)
            {
                UpdateSimulationGPU(stream);
            }
            else
            {
                UpdateSimulationCPU();
            }
        }
        void RestartSim()
        {
            assert(w > 0);
            assert(h > 0);
            FreeSim();
            InitGrid(w, h);
        }

        void ApplyForcesGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            const int edge_count_u = edges_view.edges_w_u * edges_view.edges_h_u;
            const int edge_count_v = edges_view.edges_w_v * edges_view.edges_h_v;
            const int edge_count = std::max(edge_count_u, edge_count_v);
            dim3 grid((edge_count + block.x - 1) / block.x, 1, 1);
            CodeSimulationDevice::k_apply_forces<<<grid, block, 0, stream>>>(
                edge_count_u, edge_count_v, params.wind_speed, params.g, params.velocity_dissipation, params.dt,
                cells_view, edges_view);
        }
        void DiffuseGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            const int edge_count_u = edges_view.edges_w_u * edges_view.edges_h_u;
            const int edge_count_v = edges_view.edges_w_v * edges_view.edges_h_v;
            const int edge_count = std::max(edge_count_u, edge_count_v);
            dim3 grid((edge_count + block.x - 1) / block.x, 1, 1);
            for (int i = 0; i < params.total_iter_gpu; ++i)
            {
                CodeSimulationDevice::k_diffuse<<<grid, block, 0, stream>>>(
                    edge_count_u, edge_count_v, params.viscosity, params.dt, cells_view, edges_view);
                std::swap(edges_view.u_input, edges_view.u_output);
                std::swap(edges_view.v_input, edges_view.v_output);
            }

            // edges input have the final output
        }
        void ProjectionGPU(cudaStream_t stream)
        {
            // edges input have the final output
            dim3 block(1024, 1, 1);
            dim3 grid((w * h + block.x - 1) / block.x, 1, 1);
            for (int iter = 0; iter < params.total_iter_gpu; ++iter)
            {
                CodeSimulationDevice::k_simulation_projection<<<grid, block, 0, stream>>>(
                    w * h, params.density, dx, params.dt, cells_view, edges_view);
                std::swap(cells_view.pressures_input, cells_view.pressures_output);
            }
        }
        void BeginSolidUpdate(cudaStream_t stream)
        {
            int cell_count = w * h;
            if (solid_update_requested)
            {

                CODE_API::CW_Memcpy(cells_view.solid_speeds, cells_data.solid_speeds.data(), sizeof(vec2) * cell_count,
                                    cudaMemcpyHostToDevice);
                dim3 block(1024, 1, 1);
                dim3 grid((w * h + block.x - 1) / block.x, 1, 1);
                CodeSimulationDevice::k_simulation_update_speeds_based_on_solids<<<grid, block, 0, stream>>>(
                    cell_count, cells_view, edges_view);
            }
        }
        void FinalizeSolidUpdate(cudaStream_t stream)
        {
            if (solid_update_requested)
            {
                // CopyHostToDevice(cells_view.pressures_input, cells_view.smoke_output, edges_view.u_output,
                //                  edges_view.v_output);
                solid_update_requested = false;
            }
        }

        void UpdateSimulationGPU(cudaStream_t stream)
        {
            // BeginSolidUpdate(stream);
            // vel output is stored in u_output
            ApplyForcesGPU(stream);
            // vel output is stored in u_output
            DiffuseGPU(stream);
            // vel output is stored in u_output
            ProjectionGPU(stream);
            // vel output is stored in u_output
            UpdateVelocityGPU(stream);
            // vel output is stored in u_output
            AdvectVelocityGPU(stream);
            // vel output is stored in u_output
            ProjectionGPU(stream);
            // vel output is stored in u_output
            UpdateVelocityGPU(stream);
            // vel output is stored in u_output

            // FinalizeSolidUpdate(stream);

            AdvectSmokeGPU(stream);
            DiffuseSmokeGPU(stream);

            if (params.debug)
            {
                CopyDeviceToHost(cells_view.pressures_input, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
                ProjectionResults(params.total_iter_gpu);
            }

            UpdateData();
        }


        void UpdateSimulationCPU()
        {
            AdvectVelocity();
            ApplyForces();
            Diffuse();
            Projection();
            UpdateVelocity();
            if (params.debug)
            {
                ProjectionResults(params.total_iter_cpu);
            }
            CopyHostToDevice(cells_view.pressures_input, cells_view.smoke_output, edges_view.u_output,
                             edges_view.v_output);
            UpdateData();
        }
        void Diffuse()
        {
            float a = params.viscosity * params.dt;
            float denom = 1 + 4 * a;
            for (int i = 0; i < params.total_iter_cpu; ++i)
            {
                for (int y = 0; y < edges_h_u; ++y)
                {
                    for (int x = 0; x < edges_w_u; ++x)
                    {
                        if (edges_data.GetWallU(x, y) == 0)
                        {
                            float l = edges_data.GetActiveU(x - 1, y);
                            float r = edges_data.GetActiveU(x + 1, y);
                            float b = edges_data.GetActiveU(x, y - 1);
                            float t = edges_data.GetActiveU(x, y + 1);

                            float neightbours_sum = l + r + t + b;
                            float u = (neightbours_sum)*a + edges_data.GetU(x, y);
                            edges_data.GetU(x, y) = u / denom;
                        }
                    }
                }
                for (int y = 0; y < edges_h_v; ++y)
                {
                    for (int x = 0; x < edges_w_v; ++x)
                    {
                        if (edges_data.GetWallV(x, y) == 0)
                        {
                            float l = edges_data.GetActiveV(x - 1, y);
                            float r = edges_data.GetActiveV(x + 1, y);
                            float b = edges_data.GetActiveV(x, y - 1);
                            float t = edges_data.GetActiveV(x, y + 1);

                            float neightbours_sum = l + r + t + b;
                            float v = (neightbours_sum)*a + edges_data.GetV(x, y);
                            edges_data.GetV(x, y) = v / denom;
                        }
                    }
                }
            }
        }

        void ApplyForces()
        {
            const float acceleration = params.wind_speed;

            for (int y = 0; y < edges_h_u; ++y)
            {
                for (int x = 0; x < edges_w_u; ++x)
                {
                    if (edges_data.GetWallU(x, y))
                    {
                        continue;
                    }

                    edges_data.GetU(x, y) += acceleration * params.dt;
                }
            }
        }

        void AddRadialVelocity(int x_pos, int y_pos, int radius, float scale)
        {
            if (x_pos < 0 || x_pos >= std::max(edges_w_u, edges_w_v) || y_pos < 0 ||
                y_pos >= std::max(edges_h_u, edges_h_v) || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid radial velocity parameters");
                return;
            }

            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
            const int radius_sq = radius * radius;

            for (int y = -radius; y <= radius; ++y)
            {
                for (int x = -radius; x <= radius; ++x)
                {
                    const int x_final = x_pos + x;
                    const int y_final = y_pos + y;

                    const int sq_dist = x * x + y * y;

                    if (sq_dist == 0 || sq_dist > radius_sq)
                    {
                        continue;
                    }

                    const float distance = std::sqrt(static_cast<float>(sq_dist));

                    const float u = static_cast<float>(x) / distance * scale;

                    const float v = static_cast<float>(y) / distance * scale;

                    if (x_final >= 0 && x_final < edges_w_u && y_final >= 0 && y_final < edges_h_u)
                    {
                        const int u_idx = y_final * edges_w_u + x_final;
                        if (!edges_data.is_walls_u[u_idx])
                        {
                            edges_data.u[u_idx] += u;
                        }
                    }

                    if (x_final >= 0 && x_final < edges_w_v && y_final >= 0 && y_final < edges_h_v)
                    {
                        const int v_idx = y_final * edges_w_v + x_final;
                        if (!edges_data.is_walls_v[v_idx])
                        {
                            edges_data.v[v_idx] += v;
                        }
                    }
                }
            }

            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }

        void MapImageToSmoke(int source_w, int source_h, int per_element_offset, void *data)
        {

            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }

            assert(data && "Data to map is null");

            float *data_as_float = (float *)data;

            for (int y = 0; y < cells_data.h; ++y)
            {
                for (int x = 0; x < cells_data.w; ++x)
                {
                    vec2 uv = {x / float(cells_data.w - 1), y / float(cells_data.h - 1)};
                    int source_pos_x = int(uv.x * (source_w - 1));
                    int source_pos_y = int(uv.y * (source_h - 1));
                    int base_offset = source_pos_y * source_w + source_pos_x;

                    float r = data_as_float[base_offset * per_element_offset + 0];
                    float g = data_as_float[base_offset * per_element_offset + 1];
                    float b = data_as_float[base_offset * per_element_offset + 2];
                    cells_data.smoke[y * cells_data.w + x] += vec4(r, g, b, 1.0);
                }
            }
            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }

        void MapVectorFieldUV(int source_w, int source_h, int per_element_offset, void *data)
        {

            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }

            assert(data && "Data to map is null");

            float *data_as_float = (float *)data;

            for (int y = 0; y < edges_data.edges_h_u; ++y)
            {
                for (int x = 0; x < edges_data.edges_w_u; ++x)
                {
                    vec2 uv = {float(x) / float(edges_data.edges_w_u - 1),
                               (float(y) + 0.5f) / float(edges_data.edges_h_u)};
                    int source_pos_x = int(uv.x * (source_w - 1));
                    int source_pos_y = int(uv.y * (source_h - 1));
                    int base_offset = source_pos_y * source_w + source_pos_x;

                    float u = data_as_float[base_offset * per_element_offset + 0];
                    edges_data.u[y * edges_data.edges_w_u + x] += u;
                }
            }
            for (int y = 0; y < edges_data.edges_h_v; ++y)
            {
                for (int x = 0; x < edges_data.edges_w_v; ++x)
                {
                    vec2 uv = {(float(x) + 0.5f) / float(edges_data.edges_w_v),
                               float(y) / float(edges_data.edges_h_v - 1)};
                    int source_pos_x = int(uv.x * (source_w - 1));
                    int source_pos_y = int(uv.y * (source_h - 1));
                    int base_offset = source_pos_y * source_w + source_pos_x;

                    float v = data_as_float[base_offset * per_element_offset + 1];
                    edges_data.v[y * edges_data.edges_w_v + x] += v;
                }
            }
            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }
        void AddSmoke(int x_pos, int y_pos, int radius, vec4 value)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h)
            {
                CODECUDA_PRINTLN("Invalid x,y pos");
                return;
            }
            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
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
            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }

        void AddPressure(int x_pos, int y_pos, int radius, float value)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid AddPressure parameters");
                return;
            }
            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_input, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
            for (int y = -radius; y < radius; ++y)
            {
                for (int x = -radius; x < radius; ++x)
                {
                    const int x_final = x + x_pos;
                    const int y_final = y + y_pos;
                    if (x_final < 0 || x_final >= w || y_final < 0 || y_final >= h)
                    {
                        continue;
                    }
                    const int sq_dist = x * x + y * y;
                    if (sq_dist >= radius * radius)
                    {
                        continue;
                    }
                    const int idx = y_final * w + x_final;
                    if (cells_data.is_walls[idx])
                    {
                        continue;
                    }
                    cells_data.pressures[idx] += value;
                }
            }
            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_input, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }


        void MapSolidMask(int source_w, int source_h, int *mask)
        {

            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }

            assert(mask && "Data to map is null");

            for (int y = 0; y < cells_data.h; ++y)
            {
                for (int x = 0; x < cells_data.w; ++x)
                {
                    if (x == 0 || y == 0 || x == cells_data.w - 1 || y == cells_data.h - 1)
                    {
                        continue;
                    }

                    if (x == 1 || y == 1 || x == cells_data.w - 2 || y == cells_data.h - 2)
                    {
                        continue;
                    }
                    vec2 uv = {x / float(cells_data.w - 1), y / float(cells_data.h - 1)};
                    int source_pos_x = int(uv.x * (source_w - 1));
                    int source_pos_y = int(uv.y * (source_h - 1));
                    int base_offset = source_pos_y * source_w + source_pos_x;
                    int cell_value = mask[source_pos_y * source_w + source_pos_x];
                    if (cell_value == 1)
                    {
                        continue;
                    }
                    cells_data.is_walls[y * cells_data.w + x] = cell_value;

                    int u_left;
                    int u_right;
                    int v_top;
                    int v_bottom;
                    GetCellEdgesIdxs(x, y, u_left, u_right, v_top, v_bottom);

                    edges_data.is_walls_u[u_left] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_u[u_right] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_v[v_top] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_v[v_bottom] = cells_data.is_walls[y * cells_data.w + x];
                }
            }
            for (int y = 0; y < cells_data.h; ++y)
            {
                for (int x = 0; x < cells_data.w; ++x)
                {
                    if (x == 0 || y == 0 || x == cells_data.w - 1 || y == cells_data.h - 1)
                    {
                        continue;
                    }

                    if (x == 1 || y == 1 || x == cells_data.w - 2 || y == cells_data.h - 2)
                    {
                        continue;
                    }
                    vec2 uv = {x / float(cells_data.w - 1), y / float(cells_data.h - 1)};
                    int source_pos_x = int(uv.x * (source_w - 1));
                    int source_pos_y = int(uv.y * (source_h - 1));
                    int base_offset = source_pos_y * source_w + source_pos_x;
                    int cell_value = mask[source_pos_y * source_w + source_pos_x];
                    if (cell_value == 0)
                    {
                        continue;
                    }
                    cells_data.is_walls[y * cells_data.w + x] = cell_value;

                    int u_left;
                    int u_right;
                    int v_top;
                    int v_bottom;
                    GetCellEdgesIdxs(x, y, u_left, u_right, v_top, v_bottom);

                    edges_data.is_walls_u[u_left] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_u[u_right] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_v[v_top] = cells_data.is_walls[y * cells_data.w + x];
                    edges_data.is_walls_v[v_bottom] = cells_data.is_walls[y * cells_data.w + x];
                }
            }
            UpdateCellStates();
            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }

        void SetSolidWithSpeed(int x_pos, int y_pos, vec2 vel, int radius, bool solid)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h)
            {
                CODECUDA_PRINTLN("Invalid x,y pos");
                return;
            }
            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
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
                    cells_data.is_walls[idx] = solid ? 1 : 0;
                    int x_cells = idx % w;
                    int y_cells = idx / w;

                    int u_left;
                    int u_right;
                    int v_top;
                    int v_bottom;
                    GetCellEdgesIdxs(x_cells, y_cells, u_left, u_right, v_top, v_bottom);

                    edges_data.is_walls_u[u_left] = cells_data.is_walls[idx];
                    edges_data.is_walls_u[u_right] = cells_data.is_walls[idx];
                    edges_data.is_walls_v[v_top] = cells_data.is_walls[idx];
                    edges_data.is_walls_v[v_bottom] = cells_data.is_walls[idx];
                }
            }
            UpdateCellStates();
        }
        void SetSolid(int x_pos, int y_pos, int radius, bool solid)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h)
            {
                CODECUDA_PRINTLN("Invalid x,y pos");
                return;
            }
            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
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
                    vec2 speed = vec2(x_final - x_pos, y_final - y);
                    int idx = y_final * w + x_final;
                    cells_data.is_walls[idx] = solid ? 1 : 0;
                    cells_data.solid_speeds[idx] = speed * dx;
                    int x_cells = idx % w;
                    int y_cells = idx / w;

                    int u_left;
                    int u_right;
                    int v_top;
                    int v_bottom;
                    GetCellEdgesIdxs(x_cells, y_cells, u_left, u_right, v_top, v_bottom);

                    edges_data.is_walls_u[u_left] = cells_data.is_walls[idx];
                    edges_data.is_walls_u[u_right] = cells_data.is_walls[idx];
                    edges_data.is_walls_v[v_top] = cells_data.is_walls[idx];
                    edges_data.is_walls_v[v_bottom] = cells_data.is_walls[idx];
                }
            }

            UpdateCellStates();

            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }
        void AddVelocity(int x_pos, int y_pos, int radius, float vel_x, float vel_y)
        {
            if (x_pos < 0 || x_pos >= std::max(edges_w_u, edges_w_v) || y_pos < 0 ||
                y_pos >= std::max(edges_h_u, edges_h_v))
            {
                CODECUDA_PRINTLN("Invalid x,y pos");
                return;
            }

            if (params.gpu_sim)
            {
                CopyDeviceToHost(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
            for (int y = -radius; y < radius; ++y)
            {
                for (int x = -radius; x < radius; ++x)
                {
                    int x_final = x + x_pos;
                    int y_final = y + y_pos;
                    int sq_dist = pow(x_final - x_pos, 2.0f) + pow(y_final - y_pos, 2.0f);
                    if (sq_dist >= radius * radius)
                    {
                        continue;
                    }
                    if (x_final >= 0 && x_final < edges_w_u && y_final >= 0 && y_final < edges_h_u)
                    {
                        const int u_idx = y_final * edges_w_u + x_final;
                        if (!edges_data.is_walls_u[u_idx])
                        {
                            edges_data.u[u_idx] += vel_x;
                        }
                    }

                    if (x_final >= 0 && x_final < edges_w_v && y_final >= 0 && y_final < edges_h_v)
                    {
                        const int v_idx = y_final * edges_w_v + x_final;
                        if (!edges_data.is_walls_v[v_idx])
                        {
                            edges_data.v[v_idx] += vel_y;
                        }
                    }
                }
            }

            if (params.gpu_sim)
            {
                CopyHostToDevice(cells_view.pressures_output, cells_view.smoke_output, edges_view.u_output,
                                 edges_view.v_output);
            }
        }
        void AddVelocityGPU(int x_pos, int y_pos, int radius, float vel_x, float vel_y, cudaStream_t stream)
        {
            if (x_pos < 0 || x_pos >= std::max(edges_w_u, edges_w_v) || y_pos < 0 ||
                y_pos >= std::max(edges_h_u, edges_h_v) || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid AddVelocityGPU parameters");
                return;
            }

            const int diameter = radius * 2;
            const int size = diameter * diameter;

            constexpr int threads_per_block = 256;

            const dim3 block(threads_per_block, 1, 1);
            const dim3 grid((size + threads_per_block - 1) / threads_per_block, 1, 1);

            CodeSimulationDevice::k_simulation_add_velocity<<<grid, block, 0, stream>>>(size, x_pos, y_pos, radius,
                                                                                        vel_x, vel_y, edges_view);
        }

        void AddSmokeGPU(int x_pos, int y_pos, int radius, vec4 value, cudaStream_t stream)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid AddSmoke GPU parameters");
                return;
            }

            const int diameter = radius * 2;
            const int size = diameter * diameter;

            constexpr int threads_per_block = 256;

            const dim3 block(threads_per_block, 1, 1);
            const dim3 grid((size + threads_per_block - 1) / threads_per_block, 1, 1);

            CodeSimulationDevice::k_simulation_add_smoke<<<grid, block, 0, stream>>>(size, x_pos, y_pos, radius, value,
                                                                                     cells_view);
        }

        void AddPressureGPU(int x_pos, int y_pos, int radius, float value, cudaStream_t stream)
        {
            if (x_pos < 0 || x_pos >= w || y_pos < 0 || y_pos >= h || radius <= 0)
            {
                CODECUDA_PRINTLN("Invalid AddPressureGPU parameters");
                return;
            }

            const int diameter = radius * 2;
            const int size = diameter * diameter;

            constexpr int threads_per_block = 256;

            const dim3 block(threads_per_block, 1, 1);
            const dim3 grid((size + threads_per_block - 1) / threads_per_block, 1, 1);

            CodeSimulationDevice::k_simulation_add_pressure<<<grid, block, 0, stream>>>(size, x_pos, y_pos, radius,
                                                                                        value, cells_view);
        }

    private:
        float GetVFromU(int x, int y, std::vector<float> &v_edges_old)
        {
            return SampleEdge(float(x) - 0.5f, float(y) + 0.5f, edges_w_v, edges_h_v, v_edges_old);
        }

        float GetUFromV(int x, int y, std::vector<float> &u_edges_old)
        {
            return SampleEdge(float(x) + 0.5f, float(y) - 0.5f, edges_w_u, edges_h_u, u_edges_old);
        }

        vec4 SampleSmoke(float x, float y, int cells_w, int cells_h, const std::vector<vec4> &smoke_cells)
        {

            x = std::clamp(x, 0.0f, float(cells_w - 2));
            y = std::clamp(y, 0.0f, float(cells_h - 2));
            const vec4 tl_u_prev = smoke_cells[int(y + 1) * cells_w + int(x)];
            const vec4 tr_u_prev = smoke_cells[int(y + 1) * cells_w + (int(x) + 1)];
            const vec4 bl_u_prev = smoke_cells[int(y) * cells_w + int(x)];
            const vec4 br_u_prev = smoke_cells[int(y) * cells_w + (int(x) + 1)];

            const float wx = x - floor(x);
            const float wy = y - floor(y);

            const vec4 top = code_math::lerp(tl_u_prev, tr_u_prev, wx);
            const vec4 bottom = code_math::lerp(bl_u_prev, br_u_prev, wx);
            return code_math::lerp(bottom, top, wy);
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
            std::vector<vec4> smoke_cells_old = cells_data.smoke;
            for (int y = 0; y < edges_h_u; ++y)
            {
                for (int x = 0; x < edges_w_u; ++x)
                {
                    int i = y * edges_w_u + x;
                    if (edges_data.is_walls_u[i])
                    {
                        continue;
                    }

                    float u = u_edges_old[i];
                    float v = GetVFromU(x, y, v_edges_old);
                    float pos[2] = {float(x), float(y)};
                    float x_pos = pos[0] - u * params.dt / dx;
                    float y_pos = pos[1] - v * params.dt / dy;
                    edges_data.u[i] = SampleEdge(x_pos, y_pos, edges_w_u, edges_h_u, u_edges_old);
                }
            }

            for (int y = 0; y < edges_h_v; ++y)
            {
                for (int x = 0; x < edges_w_v; ++x)
                {
                    int i = y * edges_w_v + x;
                    if (edges_data.is_walls_v[i])
                    {
                        continue;
                    }

                    float v = v_edges_old[i];

                    float u = GetUFromV(x, y, u_edges_old);

                    float pos[2] = {float(x), float(y)};
                    float x_pos = pos[0] - u * params.dt / dx;
                    float y_pos = pos[1] - v * params.dt / dy;
                    edges_data.v[i] = SampleEdge(x_pos, y_pos, edges_w_v, edges_h_v, v_edges_old);
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
                    float x_pos = pos[0] - u * params.dt / dx;
                    float y_pos = pos[1] - v * params.dt / dy;
                    cells_data.smoke[i] = SampleSmoke(x_pos, y_pos, cells_data.w, cells_data.h, smoke_cells_old);
                    if (x == w - 2)
                    {
                        // kill smoke
                        cells_data.smoke[i] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
        }
        void UpdateData()
        {
            sim_step_idx++;
            total_t += params.dt;
        }
        void AdvectSmokeGPU(cudaStream_t stream)
        {

            dim3 block(1024, 1, 1);
            dim3 grid((cells_view.w * cells_view.h + block.x - 1) / block.x, 1, 1);
            CodeSimulationDevice::k_simulation_advection_smoke<<<grid, block, 0, stream>>>(
                cells_view.w * cells_view.h, params.dt, dx, dy, params.smoke_dissipation, cells_view, edges_view);

            std::swap(cells_view.smoke_input, cells_view.smoke_output);
        }

        void DiffuseSmokeGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            dim3 grid((cells_view.w * cells_view.h + block.x - 1) / block.x, 1, 1);
            for (int i = 0; i < params.total_iter_gpu; i++)
            {
                CodeSimulationDevice::k_diffuse_smoke<<<grid, block, 0, stream>>>(
                    cells_view.w * cells_view.h, params.smoke_diffuse_coef, params.dt, cells_view, edges_view);

                std::swap(cells_view.smoke_input, cells_view.smoke_output);
            }
        }
        void AdvectVelocityGPU(cudaStream_t stream)
        {

            const int edge_count_u = edges_view.edges_w_u * edges_view.edges_h_u;
            const int edge_count_v = edges_view.edges_w_v * edges_view.edges_h_v;
            CODE_API::CW_Memcpy(edges_view.u_input, edges_view.u_output, sizeof(float) * edge_count_u,
                                cudaMemcpyDeviceToDevice);
            CODE_API::CW_Memcpy(edges_view.v_input, edges_view.v_output, sizeof(float) * edge_count_v,
                                cudaMemcpyDeviceToDevice);
            dim3 block(1024, 1, 1);
            dim3 grid_u((edge_count_u + block.x - 1) / block.x, 1, 1);
            dim3 grid_v((edge_count_v + block.x - 1) / block.x, 1, 1);
            CodeSimulationDevice::k_simulation_advection_u<<<grid_u, block, 0, stream>>>(
                edge_count_u, params.dt, dx, dy, cells_view, edges_view);
            CodeSimulationDevice::k_simulation_advection_v<<<grid_v, block, 0, stream>>>(
                edge_count_v, params.dt, dx, dy, cells_view, edges_view);
        }
        void UpdateVelocityGPU(cudaStream_t stream)
        {
            dim3 block(1024, 1, 1);
            const int edge_count_u = edges_view.edges_w_u * edges_view.edges_h_u;
            const int edge_count_v = edges_view.edges_w_v * edges_view.edges_h_v;
            dim3 grid_u((edge_count_u + block.x - 1) / block.x, 1, 1);
            dim3 grid_v((edge_count_v + block.x - 1) / block.x, 1, 1);
            float k = params.dt / (params.density * dx);
            CodeSimulationDevice::k_simulation_update_velocities_u<<<grid_u, block, 0, stream>>>(
                edge_count_u, params.dt, k, cells_view, edges_view);
            k = params.dt / (params.density * dy);
            CodeSimulationDevice::k_simulation_update_velocities_v<<<grid_v, block, 0, stream>>>(
                edge_count_v, params.dt, params.g, k, cells_view, edges_view);
        }
        void UpdateVelocity()
        {
            float k = params.dt / (params.density * dx);
            for (int y = 0; y < edges_h_u; ++y)
            {
                for (int x = 0; x < edges_w_u; ++x)
                {
                    if (edges_data.GetWallU(x, y))
                    {
                        edges_data.GetU(x, y) = 0.0f;
                        continue;
                    }

                    float press_r = cells_data.GetCellPressure(x, y);
                    float press_l = cells_data.GetCellPressure(x - 1, y);
                    edges_data.u[y * edges_w_u + x] =
                        edges_data.u[y * edges_w_u + x] - (k * (press_r - press_l));
                    edges_data.u[y * edges_w_u + x] *= 0.99;
                }
            }
            k = params.dt / (params.density * dy);
            for (int y = 0; y < edges_h_v; ++y)
            {
                for (int x = 0; x < edges_w_v; ++x)
                {
                    if (edges_data.GetWallV(x, y))
                    {
                        // edges_data.v[y * edges_w_v + x] = 0.0f;
                        continue;
                    }
                    float press_t = cells_data.GetCellPressure(x, y);
                    float press_b = cells_data.GetCellPressure(x, y - 1);
                    edges_data.v[y * edges_w_v + x] =
                        edges_data.v[y * edges_w_v + x] - (k * (press_t - press_b));
                    edges_data.v[y * edges_w_v + x] *= 0.99;
                }
            }
        }

        void Projection()
        {
            for (int iter = 0; iter < params.total_iter_cpu; ++iter)
            {
                for (int i = 0; i < cells_data.pressures.size(); ++i)
                {

                    int x = i % w;
                    int y = i / w;
                    if (cells_data.is_walls[i] == 1)
                    {
                        continue;
                    }
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

                    GetCellEdgesIdxs(x, y, edge_u_left_out_idx, edge_u_right_out_idx, edge_v_top_out_idx,
                                     edge_v_bottom_out_idx);
                    float press_l = cells_data.GetCellPressure(x - 1, y) * cells_data.GetCellFluidState(x - 1, y);
                    float press_r = cells_data.GetCellPressure(x + 1, y) * cells_data.GetCellFluidState(x + 1, y);
                    float press_t = cells_data.GetCellPressure(x, y + 1) * cells_data.GetCellFluidState(x, y + 1);
                    float press_b = cells_data.GetCellPressure(x, y - 1) * cells_data.GetCellFluidState(x, y - 1);

                    float press_sum = (press_l + press_r + press_t + press_b);

                    float u_r = edges_data.GetU(x + 1, y) * edges_data.GetStateU(x + 1, y);
                    float u_l = edges_data.GetU(x, y) * edges_data.GetStateU(x, y);
                    float v_t = edges_data.GetV(x, y + 1) * edges_data.GetStateV(x, y + 1);
                    float v_b = edges_data.GetV(x, y) * edges_data.GetStateV(x, y);

                    float velocities_sum = u_r - u_l + v_t - v_b;
                    float pressure_new =
                        (press_sum / float(s)) - (params.density * dx * velocities_sum) / (float(s) * params.dt);
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

                cells_data.divs[i] = u_r - u_l + v_t - v_b;

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

            edge_u_left_out = y * edges_w_u + x;
            edge_u_right_out = y * edges_w_u + (x + 1);

            edge_v_top_out = (y + 1) * edges_w_v + x;
            edge_v_bottom_out = y * edges_w_v + x;
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

            for (int i = 0; i < edges_data.v.size(); ++i)
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

        float dx = 0.0f;
        float dy = 0.0f;

    public:
        bool ready_to_run = false;
        int w = -1;
        int h = -1;
        int edges_w_u = -1;
        int edges_h_u = -1;
        int edges_w_v = -1;
        int edges_h_v = -1;
        c_cells_view cells_view = {};
        c_edges_view edges_view = {};
        c_cells cells_data = {};
        c_edges edges_data = {};
        int64_t sim_step_idx = 0;
        float total_t = 0.0f;
        sim_params params = {};
        bool solid_update_requested = false;
    };


} // namespace CodeCuda::FluidSimulation
#endif // CODECOMMON_HPP
