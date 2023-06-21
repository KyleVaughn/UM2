#include "../test_framework.hpp"
#include <um2/geometry/line_segment.hpp>

template <len_t D, typename T>
UM2_HOSTDEV static constexpr auto
makeLine() -> um2::LineSegment<D, T>
{
  um2::LineSegment<D, T> line;
  for (len_t i = 0; i < D; ++i) {
    line.vertices[0][i] = static_cast<T>(1);
    line.vertices[1][i] = static_cast<T>(2);
  }
  return line;
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------
template <len_t D, typename T>
UM2_HOSTDEV
TEST_CASE(accessors)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  EXPECT_TRUE(um2::isApprox(line[0], line.vertices[0]));
  EXPECT_TRUE(um2::isApprox(line[1], line.vertices[1]));
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Point<D, T> p0 = line(0);
  EXPECT_TRUE((um2::isApprox(p0, line[0])));
  um2::Point<D, T> p1 = line(1);
  EXPECT_TRUE((um2::isApprox(p1, line[1])));
  um2::Point<D, T> p05 = line(static_cast<T>(0.5));
  um2::Point<D, T> mp = um2::midpoint(p0, p1);
  EXPECT_TRUE((um2::isApprox(p05, mp)));
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Vec<D, T> j_ref = line[1] - line[0];
  um2::Vec<D, T> j0 = line.jacobian(0);
  um2::Vec<D, T> j1 = line.jacobian(1);
  EXPECT_TRUE(um2::isApprox(j0, j_ref));
  EXPECT_TRUE(um2::isApprox(j1, j_ref));
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  T len_ref = (line[1] - line[0]).norm();
  T len = length(line);
  EXPECT_NEAR(len, len_ref, static_cast<T>(1e-5));
}
// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV
TEST_CASE(bounding_box)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::AABox<D, T> box = boundingBox(line);
  EXPECT_TRUE(um2::isApprox(line[0], box.minima));
  EXPECT_TRUE(um2::isApprox(line[1], box.maxima));
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV
TEST_CASE(is_left)
{
  um2::LineSegment2<T> line = makeLine<2, T>();
  um2::Point2<T> p0 = line[0];
  um2::Point2<T> p1 = line[1];
  p0[1] -= static_cast<T>(1); // (1, 0)
  p1[1] += static_cast<T>(1); // (2, 3)
  EXPECT_FALSE(line.isLeft(p0));
  EXPECT_TRUE(line.isLeft(p1));
  p0[1] += static_cast<T>(2); // (1, 2)
  p1[1] -= static_cast<T>(2); // (2, 1)
  EXPECT_TRUE(line.isLeft(p0));
  EXPECT_FALSE(line.isLeft(p1));
  p1[1] = static_cast<T>(2.01); // (2, 2.01)
  EXPECT_TRUE(line.isLeft(p1));
}

#if UM2_ENABLE_CUDA
template <len_t D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(length, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(bounding_box, D, T);

template <typename T>
MAKE_CUDA_KERNEL(is_left, T);
#endif

template <len_t D, typename T>
TEST_SUITE(line_segment)
{
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(length, 1, 1, D, T);
  TEST_HOSTDEV(bounding_box, 1, 1, D, T);
  TEST_HOSTDEV(is_left, 1, 1, T);
}

auto
main() -> int
{
  RUN_TESTS((line_segment<2, float>));
  RUN_TESTS((line_segment<3, float>));
  RUN_TESTS((line_segment<2, double>));
  RUN_TESTS((line_segment<3, double>));
  return 0;
}
