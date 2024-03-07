#include <um2/geometry/point.hpp>

#include "../test_macros.hpp"

Float constexpr eps = castIfNot<Float>(1e-6);

template <Int D>
HOSTDEV constexpr auto
makep1() -> um2::Point<D>
{
  um2::Point<D> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<Float>(i + 1);
  }
  return v;
}

template <Int D>
HOSTDEV constexpr auto
makep2() -> um2::Point<D>
{
  um2::Point<D> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<Float>(i + 2);
  }
  return v;
}

template <Int D>
HOSTDEV
TEST_CASE(midpoint)
{
  um2::Point<D> const p1 = makep1<D>();
  um2::Point<D> const p2 = makep2<D>();
  um2::Point<D> m = um2::midpoint(p1, p2);
  Float const three_half = static_cast<Float>(3) / 2;
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(m[i], static_cast<Float>(i) + three_half, eps);
  }
}

HOSTDEV
TEST_CASE(areCCW)
{
  um2::Point2 const p1(0, 0);
  um2::Point2 const p2(1, 1);
  um2::Point2 const p3(2, -4);
  bool b = um2::areCCW(p1, p2, p3);
  ASSERT(!b);
  b = um2::areCCW(p1, p3, p2);
  ASSERT(b);
}

HOSTDEV
TEST_CASE(areApproxCCW)
{
  um2::Point2 const p1(0, 0);
  um2::Point2 const p2(1, 1);
  um2::Point2 p3(2, 2);
  bool b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(b);
  b = um2::areApproxCCW(p1, p3, p2);
  ASSERT(b);
  p3[1] -= um2::eps_distance / 2;
  b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(b);
  p3[1] -= um2::eps_distance;
  b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(!b);
}

//==============================================================================
// CUDA
//==============================================================================

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(midpoint, D);

MAKE_CUDA_KERNEL(areCCW);

MAKE_CUDA_KERNEL(areApproxCCW);
#endif

template <Int D>
TEST_SUITE(point)
{
  TEST_HOSTDEV(midpoint, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(areCCW);
    TEST_HOSTDEV(areApproxCCW);
  }
}

auto
main() -> int
{
  RUN_SUITE(point<1>);
  RUN_SUITE(point<2>);
  RUN_SUITE(point<3>);

  return 0;
}
