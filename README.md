# CodeCudaEngine

CodeCudaEngine is a weekend side project about learning CUDA matrix multiplication optimization by building the path up step by step. The goal was to write custom matmul kernels, understand why each optimization matters, and see how close a small educational implementation can get to cuBLAS.

The project is intentionally easy to follow. Most of the CUDA work lives in one source file, [core/src/CodeCuda.cu](core/src/CodeCuda.cu), so the optimization progression is visible without jumping through a large framework.

## What This Project Covers

The CUDA source includes several matmul implementations:

- naive matrix multiplication
- coalesced global-memory access
- shared-memory tiling
- 1D block tiling
- 2D block tiling
- 2D block tiling with transposed shared memory
- warp tiling with vectorized loads

The current benchmark focuses on the `warp_tilling` kernel and compares it against cuBLAS.

## Kernel Walkthrough

The kernels are meant to show the optimization path in a readable order.

### 1. Naive Matmul

The naive kernel assigns one thread to one output element of `C`. Each thread loops over `K`, loads values directly from global memory, and accumulates the dot product.

This version is simple and useful as a correctness baseline, but it does not make good use of memory bandwidth. Neighboring threads may repeatedly load the same values, and there is no shared-memory reuse.

### 2. Coalesced Global-Memory Matmul

The coalesced version keeps the same basic one-thread-per-output idea, but improves how threads map to rows and columns so global-memory accesses are more GPU-friendly.

The main goal is to make neighboring threads access neighboring memory locations where possible. This helps the GPU combine memory transactions more efficiently.

### 3. Shared-Memory Tiling

The shared-memory tiled kernel loads tiles of `A` and `B` into shared memory before doing the dot product work.

Instead of every thread repeatedly reading the same data from global memory, a block cooperatively loads a tile once, synchronizes, and then reuses that tile from much faster shared memory. This is the first major step toward higher arithmetic intensity.

### 4. 1D Block Tiling

The 1D block-tiled kernels make each thread compute more than one output value along one dimension.

This reduces overhead compared with one thread per output element and increases data reuse inside the thread. It also starts moving more work into registers, which is important for performance but has to be balanced against register pressure.

### 5. 2D Block Tiling

The 2D block-tiled kernel makes each thread compute a small `TM x TN` output tile.

This improves reuse in both the `M` and `N` dimensions. Each thread loads a small group of values from shared memory into registers, then uses them to update multiple accumulators. This is closer to the structure used by high-performance GEMM kernels.

### 6. 2D Block Tiling With Transposed Shared Memory

This version stores the `A` tile transposed in shared memory.

The reason is access pattern control. By changing how `A` is laid out in shared memory, the kernel can make later loads more convenient for the compute loop and reduce inefficient access patterns. This is a common CUDA trick: sometimes the best memory layout inside shared memory is not the same as the original global-memory layout.

### 7. Warp Tiling With Vectorized Loads

The warp-tiled kernel is the main optimized version in the benchmark.

Instead of thinking only at the block level, it divides the block tile into warp-level tiles. Each warp computes a `WM x WN` region, and each thread owns a small set of accumulator values. The kernel also uses vectorized `float4` loads to move data more efficiently into shared memory.

This version combines several important ideas:

- block-level tiling
- warp-level work decomposition
- shared-memory reuse
- register tiling
- vectorized memory loads
- multiple output values per thread

This is the version compared against cuBLAS in the benchmark plot.

## Results

The latest benchmark plot is generated from JSON saved by the CUDA library:

![CodeCuda GEMM benchmark results](autotune_logs/benchmark_results_mkn_sorted.svg)

The chart is ordered from smaller to larger problem size using `M * K * N`.

## Why I Built It

This project is mostly for learning and exploration. CUDA optimization can feel opaque when looking only at production kernels, so this repo keeps the steps explicit: start with a basic kernel, improve memory access, use shared memory, increase arithmetic intensity, and eventually move work to warp-level tiling.

If you are interested in learning CUDA or understanding how GPU matmul optimizations are built up, this project is meant to be a readable reference.

## Project Layout

- `core/`: CUDA library, public headers, kernels, benchmarking, and JSON result export
- `app/`: benchmark executable entry point (`lib_test`)
- `tools/autotune_params.py`: parameter sweep helper for tuning only
- `tools/plot_benchmark_results.py`: reads benchmark JSON and writes an SVG plot
- `autotune_logs/`: generated benchmark JSON, logs, and SVG plots
- `visualization/`: standalone browser visualization for CUDA launch geometry
- `tests/`: source-level regression tests

## Current Tuned Kernel

The active tuned kernel is `k_matmul_bt_warp_tilling`, configured through `k_auto_tunning_params`.

Current selected parameters:

```cpp
BN = 128
BM = 64
BK = 16
WN = 64
WM = 32
WNITER = 2
TN = 4
TM = 4
```

These values were selected from autotune runs on the benchmark shapes, prioritizing larger GEMM sizes.

## Build

From the project root on Windows:

```powershell
cmd.exe /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build cmake-build-debug --target lib_test"
```

## Run Benchmark

```powershell
cmake-build-debug\app\lib_test.exe
```

The benchmark currently runs square GEMM sizes from `128` through `4096`, printing custom-kernel timing, cuBLAS timing, GFLOPS, error, and pass/fail status.

## Save Benchmark JSON

Set `CODECUDA_BENCHMARK_JSON` before running the benchmark:

```powershell
$env:CODECUDA_BENCHMARK_JSON='autotune_logs\benchmark_results.json'
cmake-build-debug\app\lib_test.exe
Remove-Item Env:\CODECUDA_BENCHMARK_JSON
```

The JSON records include the matrix shape, active autotuning parameters, custom kernel `ms` and `gflops`, cuBLAS `ms` and `gflops`, accuracy, and pass status.

The public save API lives under `CodeCuda::CodeBenchmarking`:

```cpp
CodeCuda::CodeBenchmarking::C_SaveMatmulBenchmarkResultJson(path, result);
```

## Plot Results

```powershell
py -3 tools\plot_benchmark_results.py autotune_logs\benchmark_results.json --output autotune_logs\benchmark_results_mkn_sorted.svg
```

The plot script selects the best passed result for each shape and sorts the bars by `M * K * N` from smallest to largest.

## Autotune Parameters

`tools/autotune_params.py` is only for parameter tuning. It patches `k_auto_tunning_params`, rebuilds, runs the benchmark, logs the output, and restores the source by default.

```powershell
py -3 tools\autotune_params.py
```

To run one candidate:

```powershell
py -3 tools\autotune_params.py --candidate baseline_128x64_bk16
```

## Tests

Focused tests:

```powershell
py -3 -m unittest tests.test_benchmark_json tests.test_internal_logger -q
```
