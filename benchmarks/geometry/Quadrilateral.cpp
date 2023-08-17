//=============================================================================
// Findings
//=============================================================================
// For a 10x10 grid of quadrilaterals and 262144 random points:
//  containsTriangle:       113724 us = 4.3 ns/point
//  CCWShortCircuit:        124319 us = 4.7 ns/point
//  CCWNoShortCircuit:        8980 us = 0.3 ns/point
//  CUDA CCWShortCircuit:     1922 us = 0.07 ns/point
//  CUDA CCWNoShortCircuit:   1915 us = 0.07 ns/point

#include "../helpers.hpp"
#include <um2/geometry/Quadrilateral.hpp>

constexpr Size npoints = 1 << 18;
constexpr int hi = 10;

template <typename T>
auto
makeQuadGrid() -> um2::Vector<um2::Quadrilateral2<T>>
{
  Size const n = static_cast<Size>(hi);
  um2::Vector<um2::Quadrilateral2<T>> quads(n * n);
  // Create a hi x hi grid of quadrilaterals.
  for (Size x = 0; x < n; ++x) {
    for (Size y = 0; y < n; ++y) {
      um2::Point2<T> const p0(x, y);
      um2::Point2<T> const p1(x + 1, y);
      um2::Point2<T> const p2(x + 1, y + 1);
      um2::Point2<T> const p3(x, y + 1);
      um2::Quadrilateral2<T> const q(p0, p1, p2, p3);
      quads[x * n + y] = q;
    }
  }
  return quads;
}

template <typename T>
HOSTDEV constexpr auto
containsTriangle(um2::Quadrilateral2<T> const & quad, um2::Point2<T> const & p) -> bool
{
  um2::Triangle2<T> const t1(quad[0], quad[1], quad[2]);
  bool const b0 = t1.contains(p);
  um2::Triangle2<T> const t2(quad[2], quad[3], quad[0]);
  bool const b1 = t2.contains(p);
  return b0 && b1;
}

template <typename T>
HOSTDEV constexpr auto
containsCCWShortCircuit(um2::Quadrilateral2<T> const & quad, um2::Point2<T> const & p)
    -> bool
{
  return um2::areCCW(quad[0], quad[1], p) && um2::areCCW(quad[1], quad[2], p) &&
         um2::areCCW(quad[2], quad[3], p) && um2::areCCW(quad[3], quad[0], p);
}

template <typename T>
HOSTDEV constexpr auto
containsCCWNoShortCircuit(um2::Quadrilateral2<T> const & quad, um2::Point2<T> const & p)
    -> bool
{
  bool const b0 = um2::areCCW(quad[0], quad[1], p);
  bool const b1 = um2::areCCW(quad[1], quad[2], p);
  bool const b2 = um2::areCCW(quad[2], quad[3], p);
  bool const b3 = um2::areCCW(quad[3], quad[0], p);
  return b0 && b1 && b2 && b3;
}

template <typename T>
static void
triangleDecomp(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  um2::AxisAlignedBox2<T> const box({0, 0}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & q : quads) {
      i += std::count_if(points.begin(), points.end(),
                         [&q](auto const & p) { return containsTriangle(q, p); });
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
CCWShortCircuit(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  um2::AxisAlignedBox2<T> const box({0, 0}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & q : quads) {
      i += std::count_if(points.begin(), points.end(),
                         [&q](auto const & p) { return containsCCWShortCircuit(q, p); });
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
CCWNoShortCircuit(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  um2::AxisAlignedBox2<T> const box({0, 0}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  for (auto s : state) {
    int64_t i = 0;
    for (auto const & q : quads) {
      i += std::count_if(points.begin(), points.end(), [&q](auto const & p) {
        return containsCCWNoShortCircuit(q, p);
      });
    }
    benchmark::DoNotOptimize(i);
  }
}

#if UM2_USE_CUDA
template <typename T>
__global__ void
CCWShortCircuitGPUKernel(um2::Quadrilateral2<T> * quads, um2::Point2<T> * points,
                         int * results, Size nquads, Size num_points)
{
  Size const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nquads * num_points) {
    return;
  }
  Size const q = i / num_points;
  Size const p = i % num_points;
  results[i] = containsCCWShortCircuit(quads[q], points[p]);
}

template <typename T>
static void
CCWShortCircuitGPU(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  um2::Quadrilateral2<T> * d_quads;
  transferToDevice(&d_quads, quads);

  um2::AxisAlignedBox2<T> const box({0, 0}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Point2<T> * d_points;
  transferToDevice(&d_points, points);

  um2::Vector<int> results(quads.size() * points.size());
  int * d_results;
  transferToDevice(&d_results, results);

  constexpr uint32_t threads_per_block = 256;
  uint32_t const nblocks =
      (quads.size() * points.size() + threads_per_block - 1) / threads_per_block;

  for (auto s : state) {
    CCWShortCircuitGPUKernel<<<nblocks, threads_per_block>>>(d_quads, d_points, d_results,
                                                             quads.size(), points.size());
    cudaDeviceSynchronize();
  }

  cudaFree(d_quads);
  cudaFree(d_points);
  cudaFree(d_results);
}

template <typename T>
__global__ void
CCWNoShortCircuitGPUKernel(um2::Quadrilateral2<T> * quads, um2::Point2<T> * points,
                           int * results, Size nquads, Size num_points)
{
  Size const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nquads * num_points) {
    return;
  }
  Size const q = i / num_points;
  Size const p = i % num_points;
  results[i] = containsCCWNoShortCircuit(quads[q], points[p]);
}

template <typename T>
static void
CCWNoShortCircuitGPU(benchmark::State & state)
{

  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  um2::Quadrilateral2<T> * d_quads;
  transferToDevice(&d_quads, quads);

  um2::AxisAlignedBox2<T> const box({0, 0}, {hi, hi});
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::Point2<T> * d_points;
  transferToDevice(&d_points, points);

  um2::Vector<int> results(quads.size() * points.size());
  int * d_results;
  transferToDevice(&d_results, results);

  constexpr uint32_t threads_per_block = 256;
  uint32_t const nblocks =
      (quads.size() * points.size() + threads_per_block - 1) / threads_per_block;

  for (auto s : state) {
    CCWNoShortCircuitGPUKernel<<<nblocks, threads_per_block>>>(
        d_quads, d_points, d_results, quads.size(), points.size());
    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(d_results);
  }

  cudaFree(d_quads);
  cudaFree(d_points);
  cudaFree(d_results);
}
#endif

BENCHMARK_TEMPLATE(triangleDecomp, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(CCWShortCircuit, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(CCWNoShortCircuit, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#if UM2_USE_CUDA
BENCHMARK_TEMPLATE(CCWShortCircuitGPU, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(CCWNoShortCircuitGPU, float)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
