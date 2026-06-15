//
// Created by carlo on 2026-02-01.
//

#ifndef CODECOMMON_HPP
#define CODECOMMON_HPP

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
    inline void CW_PeekAtLastError()
    {
        CUDA_CHECK(cudaPeekAtLastError());
    }
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
    inline void CW_DestroyExternalMemory(cudaExternalMemory_t extMem)
    {
        CUDA_CHECK(cudaDestroyExternalMemory(extMem));
    }
    inline void CW_ImportExternalSemaphore(
        cudaExternalSemaphore_t* extSem_out,
        const cudaExternalSemaphoreHandleDesc* semHandleDesc)
    {
        CUDA_CHECK(cudaImportExternalSemaphore(extSem_out, semHandleDesc));
    }

    inline void CW_DestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
    {
        CUDA_CHECK(cudaDestroyExternalSemaphore(extSem));
    }

    inline void CW_WaitExternalSemaphoresAsync(
        const cudaExternalSemaphore_t* extSemArray,
        const cudaExternalSemaphoreWaitParams* paramsArray,
        unsigned int numExtSems,
        cudaStream_t stream)
    {
        CUDA_CHECK(cudaWaitExternalSemaphoresAsync(
            extSemArray,
            paramsArray,
            numExtSems,
            stream));
    }

    inline void CW_SignalExternalSemaphoresAsync(
        const cudaExternalSemaphore_t* extSemArray,
        const cudaExternalSemaphoreSignalParams* paramsArray,
        unsigned int numExtSems,
        cudaStream_t stream)
    {
        CUDA_CHECK(cudaSignalExternalSemaphoresAsync(
            extSemArray,
            paramsArray,
            numExtSems,
            stream));
    }
    inline void CW_StreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(stream, flags));
    }

    inline void CW_StreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority)
    {
        CUDA_CHECK(cudaStreamCreateWithPriority(stream, flags, priority));
    }

    inline void CW_StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0)
    {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, flags));
    }

    inline void CW_StreamQuery(cudaStream_t stream)
    {
        CUDA_CHECK(cudaStreamQuery(stream));
    }
    inline void CW_EventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(event, flags));
    }

    inline void CW_EventQuery(cudaEvent_t event)
    {
        CUDA_CHECK(cudaEventQuery(event));
    }
    inline void CW_MemsetAsync(void* dst, int value, size_t count, cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemsetAsync(dst, value, count, stream));
    }
    inline void CW_Memcpy2DAsync(
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream));
    }
    template <class T>
    inline void CW_MallocPitch(T** ptr, size_t* pitch, size_t widthBytes, size_t height)
    {
        CUDA_CHECK(cudaMallocPitch((void**)ptr, pitch, widthBytes, height));
    }
    inline void CW_MemPrefetchAsync(
    const void* devPtr,
    size_t count,
    int dstDevice,
    cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaMemPrefetchAsync(devPtr, count, dstDevice, stream));
    }
    inline void CW_DeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
    {
        CUDA_CHECK(cudaDeviceGetAttribute(value, attr, device));
    }
    inline void CW_CheckKernelLaunch()
    {
        CUDA_CHECK(cudaPeekAtLastError());
    }
} // namespace CODE_API

namespace Internals
{
    struct kernel_launcher
    {
        std::function<void()> kernel;
    };
    
    void add_kernel_launcher(const std::string &name, std::function<void()> kernelFunc,
                             std::map<std::string, Internals::kernel_launcher> &kernels_out)
    {
        kernel_launcher launcher;
        launcher.kernel = std::move(kernelFunc);
        kernels_out.try_emplace(name, launcher);
    }

} // namespace Internals

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
#endif // CODECOMMON_HPP
