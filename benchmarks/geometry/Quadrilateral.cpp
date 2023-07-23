// FINDINGS: The area method is faster than the CCW method, but is not as robust.
//           contains(p) is faster when we compute the CCW test of each edge first, then
//           return, instead of short-circuiting.
//           This didn't seem to effect GPU performance at all.

#include <benchmark/benchmark.h>
#include <um2/common/Vector.hpp>
#include <um2/geometry/Triangle.hpp>
#include <um2/geometry/Quadrilateral.hpp>

#include <random>

#define D 2
#define T float 
#define NPOINTS 1 << 16
#define BOX_MAX 10  

auto randomPoint() -> um2::Point<D, T> 
{
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, BOX_MAX); 
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
      p[i] = dist(rng);
  }
  return p;
}

auto makePoints(Size n) -> um2::Vector<um2::Point<D, T>> 
{
  um2::Vector<um2::Point<D, T>> points(n);
  for (auto & p : points) {
    p = randomPoint();
  }
  return points;
}

auto makeQuadGrid() -> um2::Vector<um2::Quadrilateral<D, T>> 
{
  um2::Vector<um2::Quadrilateral<D, T>> quads(BOX_MAX * BOX_MAX);
  // Create a BOX_MAX x BOX_MAX grid of quadrilaterals.
  for (Size x = 0; x < BOX_MAX; ++x) {
    for (Size y = 0; y < BOX_MAX; ++y) {
      um2::Point<D, T> p0(x, y);
      um2::Point<D, T> p1(x + 1, y);
      um2::Point<D, T> p2(x + 1, y + 1);
      um2::Point<D, T> p3(x, y + 1);
      um2::Quadrilateral<D, T> q(p0, p1, p2, p3);
      quads[x * BOX_MAX + y] = q;
    }
  }
  return quads;
}

// NOLINTBEGIN(readability-identifier-naming)
//constexpr auto
//isConvexArea(um2::Quadrilateral<D, T> const & quad) -> bool
//{
//  T const A1 = um2::Triangle<D, T>(quad[0], quad[1], quad[2]).area();
//  T const A2 = um2::Triangle<D, T>(quad[2], quad[3], quad[0]).area();
//  T const A3 = um2::Triangle<D, T>(quad[3], quad[0], quad[1]).area();
//  T const A4 = um2::Triangle<D, T>(quad[1], quad[2], quad[3]).area();
//  return um2::abs(A1 + A2 - A3 - A4) < static_cast<T>(0.01) * (A1 + A2); 
//}
//
//constexpr auto
//isConvexCCW(um2::Quadrilateral<D, T> const & quad) -> bool
//{
//  return um2::areCCW(quad[0], quad[1], quad[2]) &&
//         um2::areCCW(quad[1], quad[2], quad[3]) &&
//         um2::areCCW(quad[2], quad[3], quad[0]) &&
//         um2::areCCW(quad[3], quad[0], quad[1]);
//}
//// NOLINTEND(readability-identifier-naming)

HOSTDEV constexpr auto
containsTriangle(um2::Quadrilateral<D, T> const & quad, 
                        um2::Point<D, T> const & p) -> bool
{
  um2::Triangle<D, T> const t1(quad[0], quad[1], quad[2]);
  bool b0 = t1.contains(p);
  um2::Triangle<D, T> const t2(quad[2], quad[3], quad[0]);
  bool b1 = t2.contains(p);
  return  b0 && b1;
}
HOSTDEV constexpr auto
containsCCWShortCircuit(um2::Quadrilateral<D, T> const & quad, 
                        um2::Point<D, T> const & p) -> bool
{
  return um2::areCCW(quad[0], quad[1], p) &&
         um2::areCCW(quad[1], quad[2], p) &&
         um2::areCCW(quad[2], quad[3], p) &&
         um2::areCCW(quad[3], quad[0], p);
}

HOSTDEV constexpr auto
containsCCWNoShortCircuit(um2::Quadrilateral<D, T> const & quad, 
                        um2::Point<D, T> const & p) -> bool
{
  bool const b0 = um2::areCCW(quad[0], quad[1], p);
  bool const b1 = um2::areCCW(quad[1], quad[2], p);
  bool const b2 = um2::areCCW(quad[2], quad[3], p);
  bool const b3 = um2::areCCW(quad[3], quad[0], p);
  return b0 && b1 && b2 && b3;
}

static void TriangleDecomp(benchmark::State& state) {
  auto const quads = makeQuadGrid();
  auto const points = makePoints(static_cast<Size>(state.range(0)));
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
        i += static_cast<int>(containsTriangle(q, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

static void CCWShortCircuit(benchmark::State& state) {
  auto const quads = makeQuadGrid();
  auto const points = makePoints(static_cast<Size>(state.range(0)));
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
        i += static_cast<int>(containsCCWShortCircuit(q, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

static void CCWNoShortCircuit(benchmark::State& state) {
  auto const quads = makeQuadGrid();
  auto const points = makePoints(static_cast<Size>(state.range(0)));
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & q : quads) {
      for (auto const & p : points) {
       i += static_cast<int>(containsCCWNoShortCircuit(q, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

#if UM2_ENABLE_CUDA
__global__ void CCWShortCircuitGPUKernel(um2::Quadrilateral<D, T> * quads, 
                                         um2::Point<D, T> * points, 
                                         int * results, Size nquads, 
                                         Size npoints)
{
  Size const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nquads * npoints) {
    return;
  }
  Size const q = i / npoints;
  Size const p = i % npoints;
  results[i] = containsCCWShortCircuit(quads[q], points[p]);
}

static void CCWShortCircuitGPU(benchmark::State& state) {
  auto const quads = makeQuadGrid();
  um2::Quadrilateral<D, T> * d_quads;
  size_t const quads_size = static_cast<size_t>(quads.size()) * sizeof(um2::Quadrilateral<D, T>);
  cudaMalloc(&d_quads, quads_size);
  cudaMemcpy(d_quads, quads.data(), quads_size, cudaMemcpyHostToDevice);

  auto const points = makePoints(static_cast<Size>(state.range(0)));
  um2::Point<D, T> * d_points;
  size_t const points_size = static_cast<size_t>(points.size()) * sizeof(um2::Point<D, T>); 
  cudaMalloc(&d_points, points_size);
  cudaMemcpy(d_points, points.data(), points_size, cudaMemcpyHostToDevice);

  int * d_results;
  size_t const results_size = static_cast<size_t>(quads.size() * points.size()) * sizeof(int); 
  cudaMalloc(&d_results, results_size);
  cudaMemset(d_results, 0, results_size);

  // NOLINTNEXTLINE
  for (auto s : state) {
    uint32_t nblocks = static_cast<uint32_t>((quads.size() * points.size() + 255) / 256);
    CCWShortCircuitGPUKernel<<<nblocks, 256>>>(
        d_quads, d_points, d_results, 
        quads.size(), 
        points.size());
    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(d_results);
  }

  cudaFree(d_quads);
  cudaFree(d_points);
  cudaFree(d_results);
}

__global__ void CCWNoShortCircuitGPUKernel(um2::Quadrilateral<D, T> * quads, 
                                         um2::Point<D, T> * points, 
                                         int * results, Size nquads, 
                                         Size npoints)
{
  Size const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nquads * npoints) {
    return;
  }
  Size const q = i / npoints;
  Size const p = i % npoints;
  results[i] = containsCCWNoShortCircuit(quads[q], points[p]);
}

static void CCWNoShortCircuitGPU(benchmark::State& state) {
  auto const quads = makeQuadGrid();
  um2::Quadrilateral<D, T> * d_quads;
  size_t const quads_size = static_cast<size_t>(quads.size()) * sizeof(um2::Quadrilateral<D, T>);
  cudaMalloc(&d_quads, quads_size);
  cudaMemcpy(d_quads, quads.data(), quads_size, cudaMemcpyHostToDevice);

  auto const points = makePoints(static_cast<Size>(state.range(0)));
  um2::Point<D, T> * d_points;
  size_t const points_size = static_cast<size_t>(points.size()) * sizeof(um2::Point<D, T>); 
  cudaMalloc(&d_points, points_size);
  cudaMemcpy(d_points, points.data(), points_size, cudaMemcpyHostToDevice);

  int * d_results;
  size_t const results_size = static_cast<size_t>(quads.size() * points.size()) * sizeof(int); 
  cudaMalloc(&d_results, results_size);
  cudaMemset(d_results, 0, results_size);

  // NOLINTNEXTLINE
  for (auto s : state) {
    uint32_t nblocks = static_cast<uint32_t>((quads.size() * points.size() + 255) / 256);
    CCWNoShortCircuitGPUKernel<<<nblocks, 256>>>(
        d_quads, d_points, d_results, 
        quads.size(), 
        points.size());
    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(d_results);
  }

  cudaFree(d_quads);
  cudaFree(d_points);
  cudaFree(d_results);
}

#endif
BENCHMARK(TriangleDecomp)->RangeMultiplier(2)->Range(16, NPOINTS) 
                          ->Unit(benchmark::kMicrosecond);
BENCHMARK(CCWShortCircuit)->RangeMultiplier(2)->Range(16, NPOINTS) 
                          ->Unit(benchmark::kMicrosecond);
BENCHMARK(CCWNoShortCircuit)->RangeMultiplier(2)->Range(16, NPOINTS) 
                             ->Unit(benchmark::kMicrosecond);
#if UM2_ENABLE_CUDA
BENCHMARK(CCWShortCircuitGPU)->RangeMultiplier(2)->Range(1024, NPOINTS) 
                             ->Unit(benchmark::kMicrosecond);
BENCHMARK(CCWNoShortCircuitGPU)->RangeMultiplier(2)->Range(1024, NPOINTS) 
                                ->Unit(benchmark::kMicrosecond);
#endif
BENCHMARK_MAIN();
