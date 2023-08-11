// FINDINGS: The area method is faster than the CCW method, but is not as robust.
//           contains(p) is faster when we compute the CCW test of each edge first, then
//           return, instead of short-circuiting for some processors.
//           This didn't seem to effect GPU performance at all.

#include "../helpers.hpp"
#include <um2/geometry/Quadrilateral.hpp>

constexpr Size npoints = 1 << 18;
constexpr Size dim = 2;
constexpr int lo = 0;
constexpr int hi = 10;

// NOLINTBEGIN(readability-identifier-naming)

template <typename T>
auto
makeQuadGrid() -> um2::Vector<um2::Quadrilateral<dim, T>>
{
  um2::Vector<um2::Quadrilateral<dim, T>> quads(static_cast<Size>(hi * hi));
  // Create a hi x hi grid of quadrilaterals.
  for (Size x = 0; x < static_cast<Size>(hi); ++x) {
    for (Size y = 0; y < static_cast<Size>(hi); ++y) {
      um2::Point<dim, T> const p0(x, y);
      um2::Point<dim, T> const p1(x + 1, y);
      um2::Point<dim, T> const p2(x + 1, y + 1);
      um2::Point<dim, T> const p3(x, y + 1);
      um2::Quadrilateral<dim, T> const q(p0, p1, p2, p3);
      quads[x * static_cast<Size>(hi) + y] = q;
    }
  }
  return quads;
}

// NOLINTBEGIN(readability-identifier-naming)
// constexpr auto
// isConvexArea(um2::Quadrilateral<dim, T> const & quad) -> bool
//{
//  T const A1 = um2::Triangle<dim, T>(quad[0], quad[1], quad[2]).area();
//  T const A2 = um2::Triangle<dim, T>(quad[2], quad[3], quad[0]).area();
//  T const A3 = um2::Triangle<dim, T>(quad[3], quad[0], quad[1]).area();
//  T const A4 = um2::Triangle<dim, T>(quad[1], quad[2], quad[3]).area();
//  return um2::abs(A1 + A2 - A3 - A4) < static_cast<T>(0.01) * (A1 + A2);
//}
//
// constexpr auto
// isConvexCCW(um2::Quadrilateral<dim, T> const & quad) -> bool
//{
//  return um2::areCCW(quad[0], quad[1], quad[2]) &&
//         um2::areCCW(quad[1], quad[2], quad[3]) &&
//         um2::areCCW(quad[2], quad[3], quad[0]) &&
//         um2::areCCW(quad[3], quad[0], quad[1]);
//}
//// NOLINTEND(readability-identifier-naming)

template <typename T>
HOSTDEV constexpr auto
containsTriangle(um2::Quadrilateral<dim, T> const & quad, um2::Point<dim, T> const & p)
    -> bool
{
  um2::Triangle<dim, T> const t1(quad[0], quad[1], quad[2]);
  bool const b0 = t1.contains(p);
  um2::Triangle<dim, T> const t2(quad[2], quad[3], quad[0]);
  bool const b1 = t2.contains(p);
  return b0 && b1;
}

template <typename T>
HOSTDEV constexpr auto
containsCCWShortCircuit(um2::Quadrilateral<dim, T> const & quad, um2::Point<dim, T> const & p)
    -> bool
{
  return um2::areCCW(quad[0], quad[1], p) && um2::areCCW(quad[1], quad[2], p) &&
         um2::areCCW(quad[2], quad[3], p) && um2::areCCW(quad[3], quad[0], p);
}

template <typename T>
HOSTDEV constexpr auto
containsCCWNoShortCircuit(um2::Quadrilateral<dim, T> const & quad,
                          um2::Point<dim, T> const & p) -> bool
{
  bool const b0 = um2::areCCW(quad[0], quad[1], p);
  bool const b1 = um2::areCCW(quad[1], quad[2], p);
  bool const b2 = um2::areCCW(quad[2], quad[3], p);
  bool const b3 = um2::areCCW(quad[3], quad[0], p);
  return b0 && b1 && b2 && b3;
}

template <typename T>
static void
TriangleDecomp(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  auto const quads = makeQuadGrid<T>();
  auto const points = makeVectorOfRandomPoints<dim, T, lo, hi>(n);
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(containsTriangle(q, p));
      }
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
  auto const points = makeVectorOfRandomPoints<dim, T, lo, hi>(n);
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(containsCCWShortCircuit(q, p));
      }
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
  auto const points = makeVectorOfRandomPoints<dim, T, lo, hi>(n);
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(containsCCWNoShortCircuit(q, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

#if UM2_ENABLE_CUDA
template <typename T>
__global__ void
CCWShortCircuitGPUKernel(um2::Quadrilateral<dim, T> * quads, um2::Point<dim, T> * points,
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
  um2::Quadrilateral<dim, T> * d_quads;
  transferToDevice(&d_quads, quads);

  auto const points = makeVectorOfRandomPoints<dim, T, lo, hi>(n);
  um2::Point<dim, T> * d_points;
  transferToDevice(&d_points, points);

  um2::Vector<int> results(quads.size() * points.size());
  int * d_results;
  transferToDevice(&d_results, results);

  constexpr uint32_t threads_per_block = 256;
  uint32_t const nblocks =
      (quads.size() * points.size() + threads_per_block - 1) / threads_per_block;

  // NOLINTNEXTLINE
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
CCWNoShortCircuitGPUKernel(um2::Quadrilateral<dim, T> * quads, um2::Point<dim, T> * points,
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
  um2::Quadrilateral<dim, T> * d_quads;
  transferToDevice(&d_quads, quads);

  auto const points = makeVectorOfRandomPoints<dim, T, lo, hi>(n);
  um2::Point<dim, T> * d_points;
  transferToDevice(&d_points, points);

  um2::Vector<int> results(quads.size() * points.size());
  int * d_results;
  transferToDevice(&d_results, results);

  constexpr uint32_t threads_per_block = 256;
  uint32_t const nblocks =
      (quads.size() * points.size() + threads_per_block - 1) / threads_per_block;

  // NOLINTNEXTLINE
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

// NOLINTEND(readability-identifier-naming)
BENCHMARK_TEMPLATE(TriangleDecomp, float)
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
#if UM2_ENABLE_CUDA
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
