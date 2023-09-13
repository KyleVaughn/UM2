//=============================================================================
// Findings
//=============================================================================
// On i7-12800H, exp<double> is 2x slower than exp<float>
// The time per exp (double) is approximately 0.5ns
// This is vectorized with AVX2, so 4 exps are computed at once
// It doesn't appear that multiple threads help
// After about 100k exps, the 3050 GPU is faster than the CPU

#include "../helpers.hpp"
#include <um2/stdlib/math.hpp>

constexpr Size npoints = 1 << 20;
constexpr int lo = -3;
constexpr int hi = 3;

template <typename T>
void
expCPU(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> x =
      makeVectorOfRandomFloats<T>(n, static_cast<T>(lo), static_cast<T>(hi));
  um2::Vector<T> expx(n);
  for (auto s : state) {
    std::transform(x.begin(), x.end(), expx.begin(), um2::exp<T>);
  }
}

#if UM2_USE_TBB
template <typename T>
void
expCPUThreads(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> x =
      makeVectorOfRandomFloats<T>(n, static_cast<T>(lo), static_cast<T>(hi));
  um2::Vector<T> expx(n);
  for (auto s : state) {
    std::transform(std::execution::par_unseq, x.begin(), x.end(), expx.begin(),
                   um2::exp<T>);
  }
}
#endif

#if UM2_USE_CUDA
template <typename T>
static __global__ void
expFloatKernel(T * x, T * expx, Size const n)
{
  Size const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    expx[i] = um2::exp<T>(x[i]);
  }
}

template <typename T>
void
expFloatCUDA(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<T> x =
      makeVectorOfRandomFloats<T>(n, static_cast<T>(lo), static_cast<T>(hi));
  um2::Vector<T> expx(n);
  T * x_d;
  T * expx_d;
  transferToDevice(&x_d, x);
  transferToDevice(&expx_d, expx);

  constexpr uint32_t threadsPerBlock = 256;
  uint32_t const blocks =
      (static_cast<uint32_t>(n) + threadsPerBlock - 1) / threadsPerBlock;
  for (auto s : state) {
    expFloatKernel<<<(blocks), threadsPerBlock>>>(x_d, expx_d, n);
    cudaDeviceSynchronize();
  }
  transferFromDevice(expx, expx_d);
  cudaFree(x_d);
  cudaFree(expx_d);
}
#endif

BENCHMARK_TEMPLATE(expCPU, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(expCPU, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);

#if UM2_USE_TBB
BENCHMARK_TEMPLATE(expCPUThreads, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(expCPUThreads, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif

#if UM2_USE_CUDA
BENCHMARK_TEMPLATE(expFloatCUDA, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(expFloatCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
