#include "../test_framework.hpp"
#include <um2/geometry/point.hpp>

template <len_t D, typename T>
UM2_HOSTDEV static constexpr auto makep1() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (len_t i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <len_t D, typename T>
UM2_HOSTDEV static constexpr auto makep2() -> um2::Point<D, T>
{
  um2::Point<D, T> v;
  for (len_t i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 2);
  }
  return v;
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV TEST_CASE(distance)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  T d2 = um2::distanceSquared(p1, p2);
  if constexpr (std::floating_point<T>) {
    EXPECT_NEAR(d2, static_cast<T>(D), 1e-6);
  } else {
    EXPECT_EQ(d2, static_cast<T>(D));
  }

  T d = um2::distance(p1, p2);
  d *= d;
  if constexpr (std::floating_point<T>) {
    EXPECT_NEAR(d, static_cast<T>(D), 1e-6);
  } else {
    EXPECT_EQ(d, static_cast<T>(D));
  }
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV TEST_CASE(midpoint)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  um2::Point<D, T> m = um2::midpoint(p1, p2);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(m[i], static_cast<T>(i + 1.5), 1e-6);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(is_approx)
{
  um2::Point<D, T> p1 = makep1<D, T>();
  um2::Point<D, T> p2 = makep2<D, T>();
  // Trivial equality
  EXPECT_TRUE(um2::isApprox(p1, p1));
  // Trivial inequality
  EXPECT_FALSE(um2::isApprox(p1, p2));
  // Non-trivial equality
  p2 = p1;
  p2[0] += um2::epsilonDistance<T>() / 2;
  EXPECT_TRUE(um2::isApprox(p1, p2));
  // Non-trivial inequality
  p2[0] += um2::epsilonDistance<T>();
  EXPECT_FALSE(um2::isApprox(p1, p2));
}

template <typename T>
UM2_HOSTDEV TEST_CASE(areCCW)
{
  um2::Point2<T> p1(0, 0);
  um2::Point2<T> p2(1, 1);
  um2::Point2<T> p3(2, -4);
  bool b = um2::areCCW(p1, p2, p3);
  EXPECT_FALSE(b);
  b = um2::areCCW(p1, p3, p2);
  EXPECT_TRUE(b);
}

// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <len_t D, typename T>
MAKE_CUDA_KERNEL(distance, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(midpoint, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(is_approx, D, T);

template <typename T>
MAKE_CUDA_KERNEL(areCCW, T);
#endif

template <len_t D, typename T>
TEST_SUITE(point)
{
  TEST_HOSTDEV(distance, 1, 1, D, T);
  TEST_HOSTDEV(midpoint, 1, 1, D, T);
  TEST_HOSTDEV(is_approx, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(areCCW, 1, 1, T);
  }
}

auto main() -> int
{
  RUN_TESTS((point<1, float>));
  RUN_TESTS((point<1, double>));

  RUN_TESTS((point<2, float>));
  RUN_TESTS((point<2, double>));

  RUN_TESTS((point<3, float>));
  RUN_TESTS((point<3, double>));

  return 0;
}
