//=============================================================================
// Results
//=============================================================================
// CPU: i7-12800H
// GPU: RTX 3050
// clang-format off
// expCPU<float>/1024                  1.66 us         1.66 us       439450 bytes_per_second=2.30003Gi/s items_per_second=617.41M/s
// expCPU<float>/4096                  6.63 us         6.62 us       105706 bytes_per_second=2.30372Gi/s items_per_second=618.401M/s
// expCPU<float>/16384                 26.5 us         26.5 us        26410 bytes_per_second=2.30286Gi/s items_per_second=618.17M/s
// expCPU<float>/65536                  106 us          106 us         6609 bytes_per_second=2.29858Gi/s items_per_second=617.021M/s
// expCPU<float>/262144                 426 us          426 us         1644 bytes_per_second=2.29453Gi/s items_per_second=615.935M/s
// expCPU<float>/1048576               1714 us         1714 us          407 bytes_per_second=2.27965Gi/s items_per_second=611.938M/s
// expCPU<double>/1024                 2.86 us         2.86 us       244942 bytes_per_second=2.67162Gi/s items_per_second=358.579M/s
// expCPU<double>/4096                 11.5 us         11.5 us        60882 bytes_per_second=2.65786Gi/s items_per_second=356.732M/s
// expCPU<double>/16384                46.0 us         46.0 us        15231 bytes_per_second=2.65245Gi/s items_per_second=356.006M/s
// expCPU<double>/65536                 184 us          184 us         3808 bytes_per_second=2.65659Gi/s items_per_second=356.562M/s
// expCPU<double>/262144                736 us          736 us          951 bytes_per_second=2.65207Gi/s items_per_second=355.955M/s
// expCPU<double>/1048576              2982 us         2982 us          234 bytes_per_second=2.61966Gi/s items_per_second=351.605M/s
// expCPUThreads<float>/1024           14.8 us         6.51 us       105338 bytes_per_second=599.758Mi/s items_per_second=157.223M/s
// expCPUThreads<float>/4096           16.1 us         7.19 us       107113 bytes_per_second=2.12141Gi/s items_per_second=569.461M/s
// expCPUThreads<float>/16384          19.5 us         9.45 us        85071 bytes_per_second=6.46051Gi/s items_per_second=1.73423G/s
// expCPUThreads<float>/65536          31.2 us         18.3 us        39942 bytes_per_second=13.3073Gi/s items_per_second=3.57216G/s
// expCPUThreads<float>/262144         77.2 us         56.1 us        10657 bytes_per_second=17.4043Gi/s items_per_second=4.67193G/s
// expCPUThreads<float>/1048576         264 us          219 us         2965 bytes_per_second=17.8185Gi/s items_per_second=4.78311G/s
// expCPUThreads<double>/1024          15.9 us         6.84 us       104072 bytes_per_second=1.11459Gi/s items_per_second=149.598M/s
// expCPUThreads<double>/4096          17.6 us         7.94 us        93769 bytes_per_second=3.8422Gi/s items_per_second=515.692M/s
// expCPUThreads<double>/16384         23.3 us         12.8 us        54576 bytes_per_second=9.54043Gi/s items_per_second=1.28049G/s
// expCPUThreads<double>/65536         46.2 us         32.3 us        20273 bytes_per_second=15.0945Gi/s items_per_second=2.02595G/s
// expCPUThreads<double>/262144         135 us          110 us         6141 bytes_per_second=17.7113Gi/s items_per_second=2.37717G/s
// expCPUThreads<double>/1048576        503 us          433 us         1651 bytes_per_second=18.0374Gi/s items_per_second=2.42094G/s
// expCUDA<float>/65536                6.62 us         6.62 us        91202 bytes_per_second=36.8608Gi/s items_per_second=9.89474G/s
// expCUDA<float>/262144               18.1 us         18.1 us        38792 bytes_per_second=53.9043Gi/s items_per_second=14.4698G/s
// expCUDA<float>/1048576              56.5 us         56.5 us        12414 bytes_per_second=69.1308Gi/s items_per_second=18.5572G/s
// expCUDA<double>/65536               25.7 us         25.7 us        27247 bytes_per_second=18.9924Gi/s items_per_second=2.54911G/s
// expCUDA<double>/262144              89.7 us         89.7 us         7794 bytes_per_second=21.7739Gi/s items_per_second=2.92245G/s
// expCUDA<double>/1048576              342 us          342 us         2048 bytes_per_second=22.8677Gi/s items_per_second=3.06925G/s
// clang-format on

#include "../helpers.hpp"

#include <um2/config.hpp>
#include <um2/stdlib/math/exponential_functions.hpp>
#include <um2/stdlib/vector.hpp>

#include <benchmark/benchmark.h>

#if UM2_USE_CUDA
#  include <um2/common/cast_if_not.hpp>
#  include <um2/stdlib/math/abs.hpp>
#endif

Int constexpr npoints = 1 << 20;
Int constexpr lo = -10;
Int constexpr hi = 10;

template <class T>
void
expCPU(benchmark::State & state)
{
  Int const n = static_cast<Int>(state.range(0));
  um2::Vector<T> x =
      makeVectorOfRandomFloats<T>(n, static_cast<T>(lo), static_cast<T>(hi));
  um2::Vector<T> expx(n);
  for (auto s : state) {
    for (Int i = 0; i < n; ++i) {
      expx[i] = um2::exp<T>(x[i]);
    }
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * n * static_cast<Int>(sizeof(T)));
}

#if UM2_USE_OPENMP
template <typename T>
void
expCPUThreads(benchmark::State & state)
{
  Int const n = static_cast<Int>(state.range(0));
  um2::Vector<T> x =
      makeVectorOfRandomFloats<T>(n, static_cast<T>(lo), static_cast<T>(hi));
  um2::Vector<T> expx(n);
  for (auto s : state) {
#  pragma omp parallel for
    for (Int i = 0; i < n; ++i) {
      expx[i] = um2::exp<T>(x[i]);
    }
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * n * static_cast<Int>(sizeof(T)));
}
#endif

#if UM2_USE_CUDA
template <class T>
static __global__ void
expFloatKernel(T * x, T * expx, Int const n)
{
  Int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    expx[i] = um2::exp(x[i]);
  }
}

template <class T>
void
expCUDA(benchmark::State & state)
{
  Int const n = static_cast<Int>(state.range(0));
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
  // Ensure that the computed values are correct
  T constexpr eps = castIfNot<T>(1e-3);
  for (Int i = 0; i < n; ++i) {
    // Get the relative error
    T const abs_rel_err = um2::abs((expx[i] - um2::exp(x[i])) / um2::exp(x[i]));
    ASSERT(abs_rel_err < eps);
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * n * static_cast<Int>(sizeof(T)));
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

#if UM2_USE_OPENMP
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
BENCHMARK_TEMPLATE(expCUDA, float)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(expCUDA, double)
    ->RangeMultiplier(4)
    ->Range(65536, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
