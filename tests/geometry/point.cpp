#include <um2/geometry/point.hpp>

#include "../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

template <Size D>
HOSTDEV constexpr auto
makep1() -> um2::Point<D>
{
  um2::Point<D> v;
  for (Size i = 0; i < D; ++i) {
    v[i] = static_cast<F>(i + 1);
  }
  return v;
}

template <Size D>
HOSTDEV constexpr auto
makep2() -> um2::Point<D>
{
  um2::Point<D> v;
  for (Size i = 0; i < D; ++i) {
    v[i] = static_cast<F>(i + 2);
  }
  return v;
}

template <Size D>
HOSTDEV
TEST_CASE(distance)
{
  um2::Point<D> const p1 = makep1<D>();
  um2::Point<D> const p2 = makep2<D>();
  F const d2 = p1.squaredDistanceTo(p2);
  ASSERT_NEAR(d2, D, eps); 

  F d = p1.distanceTo(p2);
  d *= d;
  ASSERT_NEAR(d, static_cast<F>(D), eps); 
}

template <Size D>
HOSTDEV
TEST_CASE(midpoint)
{
  um2::Point<D> const p1 = makep1<D>();
  um2::Point<D> const p2 = makep2<D>();
  um2::Point<D> m = um2::midpoint(p1, p2);
  F const three_half = static_cast<F>(3) / 2;
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(m[i], static_cast<F>(i) + three_half, eps);
  }
}

template <Size D>
HOSTDEV
TEST_CASE(isApprox)
{
  um2::Point<D> const p1 = makep1<D>();
  um2::Point<D> p2 = makep2<D>();
  // Trivial equality
  ASSERT(um2::isApprox(p1, p1));
  // Trivial inequality
  ASSERT(!um2::isApprox(p1, p2));
  // Non-trivial equality
  p2 = p1;
  p2[0] += um2::eps_distance / 2;
  ASSERT(um2::isApprox(p1, p2));
  // Non-trivial inequality
  p2[0] += um2::eps_distance;
  ASSERT(!um2::isApprox(p1, p2));
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
template <Size D>
MAKE_CUDA_KERNEL(distance, D);

template <Size D>
MAKE_CUDA_KERNEL(midpoint, D);

template <Size D>
MAKE_CUDA_KERNEL(isApprox, D);

MAKE_CUDA_KERNEL(areCCW);

MAKE_CUDA_KERNEL(areApproxCCW);
#endif

template <Size D>
TEST_SUITE(point)
{
  TEST_HOSTDEV(distance, 1, 1, D);
  TEST_HOSTDEV(midpoint, 1, 1, D);
  TEST_HOSTDEV(isApprox, 1, 1, D);
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
