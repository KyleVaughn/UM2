#include <um2/ray_casting/intersect/Ray-LineSegment.hpp>

#include <um2/math/math_functions.hpp>

#include "../../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(intersect)
{
  um2::LineSegment<2, T> l(um2::Point2<T>(0, 1), um2::Point2<T>(2, -1));
  um2::Vec2<T> const dir = um2::Point2<T>(1, 1).normalized();
  um2::Ray2<T> const ray(um2::Point2<T>(0, -1), dir);
  T res = 0;
  res = intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<T>(2)), static_cast<T>(1e-4));

  l = um2::LineSegment<2, T>(um2::Point2<T>(1, -1), um2::Point2<T>(1, 1));
  res = intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<T>(2)), static_cast<T>(1e-4));
}

#if UM2_ENABLE_CUDA

template <typename T>
MAKE_CUDA_KERNEL(intersect, T)

#endif

template <typename T>
TEST_SUITE(ray2_line_segment2)
{
  TEST_HOSTDEV(intersect, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE((ray2_line_segment2<float>));
  RUN_SUITE((ray2_line_segment2<double>));
  return 0;
}
