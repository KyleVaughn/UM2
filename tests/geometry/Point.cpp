#include <um2/geometry/Point.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makep1() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (Size i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makep2() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (Size i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 2);
  }
  return v;
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(distance)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  T d2 = um2::distanceSquared(p1, p2);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(d2, static_cast<T>(D), static_cast<T>(1e-6));
  } else {
    ASSERT(d2 == static_cast<T>(D));
  }

  T d = um2::distance(p1, p2);
  d *= d;
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(d, static_cast<T>(D), static_cast<T>(1e-6));
  } else {
    ASSERT(d == static_cast<T>(D));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(midpoint)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  um2::Point<D, T> m = um2::midpoint(p1, p2);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(m[i], static_cast<T>(i + 1.5), static_cast<T>(1e-6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(isApprox)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  // Trivial equality
  ASSERT(um2::isApprox(p1, p1));
  // Trivial inequality
  ASSERT(!um2::isApprox(p1, p2));
  // Non-trivial equality
  p2 = p1;
  p2[0] += um2::epsilonDistance<T>() / 2;
  ASSERT(um2::isApprox(p1, p2));
  // Non-trivial inequality
  p2[0] += um2::epsilonDistance<T>();
  ASSERT(!um2::isApprox(p1, p2));
}

template <typename T>
HOSTDEV
TEST_CASE(areCCW)
{
  um2::Point2<T> p1(0, 0);
  um2::Point2<T> p2(1, 1);
  um2::Point2<T> p3(2, -4);
  bool b = um2::areCCW(p1, p2, p3);
  ASSERT(!b);
  b = um2::areCCW(p1, p3, p2);
  ASSERT(b);
}

// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(distance, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(midpoint, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(isApprox, D, T);

template <typename T>
MAKE_CUDA_KERNEL(areCCW, T);
#endif

template <Size D, typename T>
TEST_SUITE(point)
{
  TEST_HOSTDEV(distance, 1, 1, D, T);
  TEST_HOSTDEV(midpoint, 1, 1, D, T);
  TEST_HOSTDEV(isApprox, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(areCCW, 1, 1, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((point<1, float>));
  RUN_SUITE((point<1, double>));

  RUN_SUITE((point<2, float>));
  RUN_SUITE((point<2, double>));

  RUN_SUITE((point<3, float>));
  RUN_SUITE((point<3, double>));

  return 0;
}
