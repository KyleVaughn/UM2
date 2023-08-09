// FINDINGS: 
//  The barycentric method is faster than the CCW method on some 
//  processors, but not others

#include "../helpers.hpp" 
#include <um2/geometry/Triangle.hpp>

constexpr Size npoints = 1 << 16;
constexpr Size ntris = 1 << 5;
constexpr Size D = 2;
constexpr int lo = 0;
constexpr int hi = 100;

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
constexpr auto
triContainsBary(um2::Triangle<D, T> const & tri, um2::Point<D, T> const & p) -> bool
{
  um2::Vec<D, T> const A = tri[1] - tri[0];
  um2::Vec<D, T> const B = tri[2] - tri[0];
  um2::Vec<D, T> const C = p - tri[0];
  T const invdetAB = 1 / A.cross(B);
  T const r = C.cross(B) * invdetAB;
  T const s = A.cross(C) * invdetAB;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
}

template <typename T>
constexpr auto
triContainsCCW(um2::Triangle<D, T> const & tri, um2::Point<D, T> const & p) -> bool
{
  return um2::areCCW(tri[0], tri[1], p) && um2::areCCW(tri[1], tri[2], p) &&
         um2::areCCW(tri[2], tri[0], p);
}

template <typename T>
constexpr auto
triContainsNoShortCCW(um2::Triangle<D, T> const & tri, um2::Point<D, T> const & p) -> bool
{
  bool const b0 = um2::areCCW(tri[0], tri[1], p);
  bool const b1 = um2::areCCW(tri[1], tri[2], p);
  bool const b2 = um2::areCCW(tri[2], tri[0], p);
  return b0 && b1 && b2;
}

// NOLINTEND(readability-identifier-naming)

template <typename T>
static void
containsBary(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<D, T>> const points = makeVectorOfRandomPoints<D, T, lo, hi>(n);
  um2::Vector<um2::Triangle<D, T>> const tris = makeVectorOfRandomTriangles<T, lo, hi>(ntris); 
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & t : tris) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(triContainsBary(t, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
containsCCW(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<D, T>> const points = makeVectorOfRandomPoints<D, T, lo, hi>(n);
  um2::Vector<um2::Triangle<D, T>> const tris = makeVectorOfRandomTriangles<T, lo, hi>(ntris); 
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & t : tris) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(triContainsCCW(t, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

template <typename T>
static void
containsNoShortCCW(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::Vector<um2::Point<D, T>> const points = makeVectorOfRandomPoints<D, T, lo, hi>(n);
  um2::Vector<um2::Triangle<D, T>> const tris = makeVectorOfRandomTriangles<T, lo, hi>(ntris); 
  // NOLINTNEXTLINE
  for (auto s : state) {
    int i = 0;
    for (auto const & t : tris) {
      for (auto const & p : points) {
        // cppcheck-suppress useStlAlgorithm
        i += static_cast<int>(triContainsNoShortCCW(t, p));
      }
    }
    benchmark::DoNotOptimize(i);
  }
}

BENCHMARK_TEMPLATE(containsBary, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsCCW, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(containsNoShortCCW, double)
    ->RangeMultiplier(4)
    ->Range(1024, npoints)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
