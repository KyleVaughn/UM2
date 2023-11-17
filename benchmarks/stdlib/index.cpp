//=============================================================================
// Summary
//=============================================================================
//
// Purpose:
// -------
// This benchmark aims to determine the performance difference of using a 32-bit
// vs 64-bit index. This is primarily a concern on GPUs, where 64-bit integer
// arithmetic is reported to be much slower than 32-bit integer arithmetic.
// This is important for general use, but also for the implementation of
// various mesh data structures, which often use 64-bit integers to index
// vertices and faces.
//
// Description:
// ------------
// To test this, we use a simple arithmetic kernel that performs a non-trivial
// amount of arithmetic on a vector of random integers. Typical index mapping
// calculations will lkely use less arithmetic than this, so the actual impact
// of using 64-bit indices will be less than what is reported here.
//
// Results:
// --------
// arithmeticCPU<uint32_t>/131072         351 us          351 us         1963
// arithmeticCPU<uint64_t>/131072         366 us          366 us         1900
// arithmeticCPU<int32_t>/131072          359 us          359 us         1956
// arithmeticCPU<int64_t>/131072          399 us          399 us         1754
// arithmeticCUDA<uint32_t>/131072        202 us          202 us         2705
// arithmeticCUDA<uint64_t>/131072        342 us          342 us         2048
// arithmeticCUDA<int32_t>/131072         208 us          208 us         3372
// arithmeticCUDA<int64_t>/131072         367 us          367 us         1910
//
// Analysis:
// ---------
// Statistics:
//    The sample size is relatively small, and cuda synchronization is likely
//    a source of noise. I would take these results +/- 3%.
//
// Signed vs unsigned:
//    Unsigned integers were about 2%-9% faster than signed integers on CPU and
//    GPU. This is to be expected, since unsigned arithmetic is more efficient.
//
// 32-bit vs 64-bit:
//    On CPU, 64-bit integers were about 4%-11% slower than 32-bit integers.
//    On GPU, 64-bit integers were about 69%-76% slower than 32-bit integers.
//    On x86, we expect 64-bit integer arithmetic to be approximately as fast as
//    32-bit integer arithmetic. This means that the increased memory bandwidth
//    required to load/store 64-bit integers plays a non-trivial role in the
//    performance difference between 32-bit and 64-bit integers. However,
//    I would expect this different to be less that the 69%-76% reported here.
//
// Conclusions:
// ------------
// 1. Unsigned integers are faster than signed integers, but not fast enough
//    to make me want to deal with the hassle of unsigned integers.
//
// 2. 64-bit integers are a bit slower on CPU, but much slower on GPU. Therefore
//    we should use 32-bit integers for GPU indices. Since most of the code
//    is shared between CPU and GPU, we should use 32-bit integers for CPU
//    indices as well.

#include "../helpers.hpp"

constexpr Size vals = 1 << 17;

template <typename T>
HOSTDEV auto
mix(T a, T b, T c) -> T
{
  // Some hash function ripped from the internet + a division at the end.
  // We're just trying to do a non-trivial amount of arithmetic.
  a = a - b;
  a = a - c;
  a = a ^ (c >> 13);
  b = b - c;
  b = b - a;
  b = b ^ (a << 8);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 13);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 12);
  b = b - c;
  b = b - a;
  b = b ^ (a << 16);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 5);
  a = a - b;
  a = a - c;
  a = a ^ (c >> 3);
  b = b - c;
  b = b - a;
  b = b ^ (a << 10);
  c = c - a;
  c = c - b;
  c = c ^ (b >> 15);
  return c / a;
}

template <typename T>
void
arithmeticCPU(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> x = makeVectorOfRandomInts<T>(n);
  um2::Vector<T> y = makeVectorOfRandomInts<T>(n);
  um2::Vector<T> z = makeVectorOfRandomInts<T>(n);
  for (auto s : state) {
    for (Size i = 0; i < n; ++i) {
      x[i] = mix(x[i], y[i], z[i]);
    }
  }
}

#if UM2_USE_CUDA
template <typename T>
static __global__ void
arithmeticKernel(T * x, T * y, T * z, Size const n)
{
  // Each thread is responsible for n/tpb elements.
  Size const t = threadIdx.x;
  Size const tpb = blockDim.x;
  for (Size i = t; i < n; i += tpb) {
    x[i] = mix(x[i], y[i], z[i]);
  }
}

template <typename T>
void
arithmeticCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> x = makeVectorOfRandomInts<T>(n);
  um2::Vector<T> y = makeVectorOfRandomInts<T>(n);
  um2::Vector<T> z = makeVectorOfRandomInts<T>(n);
  T * x_d;
  T * y_d;
  T * z_d;
  size_t const size_in_bytes = static_cast<size_t>(n) * sizeof(T);
  cudaMalloc(&x_d, size_in_bytes);
  cudaMalloc(&y_d, size_in_bytes);
  cudaMalloc(&z_d, size_in_bytes);
  cudaMemcpy(x_d, x.data(), size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y.data(), size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z.data(), size_in_bytes, cudaMemcpyHostToDevice);
  constexpr uint32_t tpb = 256; // threads per block
  for (auto s : state) {
    arithmeticKernel<<<1, tpb>>>(x_d, y_d, z_d, n);
    cudaDeviceSynchronize();
  }
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
}
#endif

BENCHMARK_TEMPLATE(arithmeticCPU, uint32_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCPU, uint64_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCPU, int32_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCPU, int64_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
#if UM2_USE_CUDA
BENCHMARK_TEMPLATE(arithmeticCUDA, uint32_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCUDA, uint64_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCUDA, int32_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(arithmeticCUDA, int64_t)
    ->Range(vals, vals)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
