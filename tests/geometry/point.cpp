#include <um2/config.hpp>
#include <um2/geometry/point.hpp>

#include "../test_macros.hpp"

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makep1() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <Int D, class T>
HOSTDEV constexpr auto
makep2() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 2);
  }
  return v;
}

template <Int D, class T>
HOSTDEV
TEST_CASE(midpoint)
{
  um2::Point<D, T> const p1 = makep1<D, T>();
  um2::Point<D, T> const p2 = makep2<D, T>();
  um2::Point<D, T> m = um2::midpoint(p1, p2);
  T const three_half = static_cast<T>(3) / 2;
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(m[i], static_cast<T>(i) + three_half, eps<T>);
  }
}

template <class T>
HOSTDEV
TEST_CASE(areCCW)
{
  um2::Point2<T> const p1(0, 0);
  um2::Point2<T> const p2(1, 1);
  um2::Point2<T> const p3(2, -4);
  bool b = um2::areCCW(p1, p2, p3);
  ASSERT(!b);
  b = um2::areCCW(p1, p3, p2);
  ASSERT(b);
}

template <class T>
HOSTDEV
TEST_CASE(areApproxCCW)
{
  um2::Point2<T> const p1(0, 0);
  um2::Point2<T> const p2(1, 1);
  um2::Point2<T> p3(2, 2);
  bool b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(b);
  b = um2::areApproxCCW(p1, p3, p2);
  ASSERT(b);
  p3[1] -= um2::epsDistance<T>() / 2;
  b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(b);
  p3[1] -= um2::epsDistance<T>();
  b = um2::areApproxCCW(p1, p2, p3);
  ASSERT(!b);
}

//==============================================================================
// CUDA
//==============================================================================

#if UM2_USE_CUDA
template <Int D, class T>
MAKE_CUDA_KERNEL(midpoint, D, T);

template <class T>
MAKE_CUDA_KERNEL(areCCW, T);

template <class T>
MAKE_CUDA_KERNEL(areApproxCCW, T);
#endif

template <Int D, class T>
TEST_SUITE(point)
{
  TEST_HOSTDEV(midpoint, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(areCCW, T);
    TEST_HOSTDEV(areApproxCCW, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((point<1, float>));
  RUN_SUITE((point<2, float>));
  RUN_SUITE((point<3, float>));

  RUN_SUITE((point<1, double>));
  RUN_SUITE((point<2, double>));
  RUN_SUITE((point<3, double>));
  return 0;
}
