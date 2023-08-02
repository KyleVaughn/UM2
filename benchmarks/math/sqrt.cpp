// FINDINGS:
// sqrt<double> is couple of times slower than sqrt<float>
// Multi-threaded sqrt seems to be faster after about 30k elements
// CUDA sqrt seems faster even before 30k elements
#include "../helpers.hpp"
#include <um2/math/math_functions.hpp>

constexpr Size npoints = 1 << 20;

template <typename T>
static void
sqrtCPU(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> const x = makeVectorOfRandomFloats<T, 1, 20000>(n);
  um2::Vector<T> sqrtx(n);
  // NOLINTNEXTLINE
  for (auto s : state) {
    std::transform(x.begin(), x.end(), sqrtx.begin(), um2::sqrt<T>);
  }
}

#if UM2_ENABLE_OPENMP
template <typename T>
static void
sqrtCPUThreads(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> const x = makeVectorOfRandomFloats<T, -3, 3>(n);
  um2::Vector<T> sqrtx(n);
  // NOLINTNEXTLINE
  for (auto s : state) {
    __gnu_parallel::transform(x.begin(), x.end(), sqrtx.begin(), um2::sqrt<T>);
  }
}
#endif

#if UM2_ENABLE_CUDA
template <typename T>
static __global__ void
sqrtFloatKernel(T * x, T * sqrtx, Size const n)
{
  Size const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    sqrtx[i] = um2::sqrt<T>(x[i]);
  }
}

template <typename T>
static void
sqrtFloatCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> const x = makeVectorOfRandomFloats<T, -3, 3>(n);
  um2::Vector<T> sqrtx(n);
  T * x_d;
  T * sqrtx_d;
  transferToDevice(&x_d, x);
  transferToDevice(&sqrtx_d, sqrtx);

  constexpr int threadsPerBlock = 256;
  // NOLINTNEXTLINE
  for (auto s : state) {
    sqrtFloatKernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        x_d, sqrtx_d, n);
    cudaDeviceSynchronize();
  }
  transferFromDevice(sqrtx, sqrtx_d);
  cudaFree(x_d);
  cudaFree(sqrtx_d);
}
#endif

BENCHMARK_TEMPLATE(sqrtCPU, float)->RangeMultiplier(2)->Range(1024, npoints);
BENCHMARK_TEMPLATE(sqrtCPU, double)->RangeMultiplier(2)->Range(1024, npoints);

#if UM2_ENABLE_OPENMP
BENCHMARK_TEMPLATE(sqrtCPUThreads, float)->RangeMultiplier(2)->Range(1024, npoints);
BENCHMARK_TEMPLATE(sqrtCPUThreads, double)->RangeMultiplier(2)->Range(1024, npoints);
#endif

#if UM2_ENABLE_CUDA
BENCHMARK_TEMPLATE(sqrtFloatCUDA, float)->RangeMultiplier(2)->Range(65536, npoints);
BENCHMARK_TEMPLATE(sqrtFloatCUDA, double)->RangeMultiplier(2)->Range(65536, npoints);
#endif
BENCHMARK_MAIN();
