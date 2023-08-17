#include <um2/geometry/LineSegment.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeLine() -> um2::LineSegment<D, T>
{
  um2::LineSegment<D, T> line;
  for (Size i = 0; i < D; ++i) {
    line.v[0][i] = static_cast<T>(1);
    line.v[1][i] = static_cast<T>(2);
  }
  return line;
}

//=============================================================================
// Accessors
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  ASSERT(um2::isApprox(line[0], line.v[0]));
  ASSERT(um2::isApprox(line[1], line.v[1]));
}

//=============================================================================
// Interpolation
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Point<D, T> const p0 = line(0);
  ASSERT((um2::isApprox(p0, line[0])));
  um2::Point<D, T> const p1 = line(1);
  ASSERT((um2::isApprox(p1, line[1])));
  um2::Point<D, T> const p05 = um2::interpolate(line, static_cast<T>(0.5));
  um2::Point<D, T> const mp = um2::midpoint(p0, p1);
  ASSERT((um2::isApprox(p05, mp)));
}

//=============================================================================
// jacobian
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Vec<D, T> j_ref;
  for (Size i = 0; i < D; ++i) {
    j_ref[i] = line[1][i] - line[0][i];
  }
  um2::Vec<D, T> const j0 = line.jacobian(0);
  um2::Vec<D, T> const j1 = um2::jacobian(line, 1);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));
}

//=============================================================================
// getRotation
//=============================================================================

template <typename T>
HOSTDEV
TEST_CASE(getRotation)
{
  um2::LineSegment<2, T> line = makeLine<2, T>();
  um2::Mat2x2<T> rot = line.getRotation();
  um2::LineSegment<2, T> line_rot(rot * line[0], rot * line[1]);
  ASSERT_NEAR(line_rot[0][1], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(line_rot[1][1], static_cast<T>(0), static_cast<T>(1e-5));
  um2::LineSegment<2, T> line_rot2(um2::zeroVec<2, T>(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot2[0][0], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(line_rot2[0][1], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(line_rot2[1][1], static_cast<T>(0), static_cast<T>(1e-5));
  line[0][0] = static_cast<T>(10);
  rot = line.getRotation();
  um2::LineSegment<2, T> line_rot3(um2::zeroVec<2, T>(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot3[0][0], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(line_rot3[0][1], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(line_rot3[1][1], static_cast<T>(0), static_cast<T>(1e-5));
}

//=============================================================================
// length
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  T len_ref = line[0].distanceTo(line[1]);
  ASSERT_NEAR(line.length(), len_ref, static_cast<T>(1e-5));
  ASSERT_NEAR(um2::length(line), len_ref, static_cast<T>(1e-5));
}

//=============================================================================
// boundingBox
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::AxisAlignedBox<D, T> const box = line.boundingBox();
  ASSERT(um2::isApprox(line[0], box.minima));
  ASSERT(um2::isApprox(line[1], box.maxima));
  um2::AxisAlignedBox<D, T> const box2 = um2::boundingBox(line);
  ASSERT(um2::isApprox(box2.minima, box.minima));
  ASSERT(um2::isApprox(box2.maxima, box.maxima));
}

//=============================================================================
// isLeft
//=============================================================================

template <typename T>
HOSTDEV
TEST_CASE(isLeft)
{
  um2::LineSegment2<T> line = makeLine<2, T>();
  um2::Point2<T> p0 = line[0];
  um2::Point2<T> p1 = line[1];
  p0[1] -= static_cast<T>(1); // (1, 0)
  p1[1] += static_cast<T>(1); // (2, 3)
  ASSERT(!line.isLeft(p0));
  ASSERT(line.isLeft(p1));
  p0[1] += static_cast<T>(2); // (1, 2)
  p1[1] -= static_cast<T>(2); // (2, 1)
  ASSERT(line.isLeft(p0));
  ASSERT(!line.isLeft(p1));
  p1[1] = static_cast<T>(2.01); // (2, 2.01)
  ASSERT(line.isLeft(p1));
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(distanceTo)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();

  // The left end point
  um2::Point<D, T> p0 = line[0];
  ASSERT_NEAR(line.distanceTo(p0), static_cast<T>(0), static_cast<T>(1e-5));
  // A point to the left of the left end point
  p0[0] -= static_cast<T>(1);
  ASSERT_NEAR(line.distanceTo(p0), static_cast<T>(1), static_cast<T>(1e-5));
  // A point to the right of the left end point
  p0[0] += static_cast<T>(1.5);
  T ref = 0;
  if constexpr (D == 2) {
    ref = um2::sin(um2::pi<T>() / static_cast<T>(4)) / 2;
  } else {
    // d = (7/6, 1/6, 1/6)
    ref = um2::sqrt(static_cast<T>(6)) / 6;
  }
  ASSERT_NEAR(line.distanceTo(p0), ref, static_cast<T>(1e-5));

  // Repeat for the right end point
  um2::Point<D, T> p1 = line[1];
  ASSERT_NEAR(line.distanceTo(p1), static_cast<T>(0), static_cast<T>(1e-5));
  p1[0] += static_cast<T>(1);
  ASSERT_NEAR(line.distanceTo(p1), static_cast<T>(1), static_cast<T>(1e-5));
  p1[0] -= static_cast<T>(1.5);
  ASSERT_NEAR(line.distanceTo(p1), ref, static_cast<T>(1e-5));
}

#if UM2_USE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <typename T>
MAKE_CUDA_KERNEL(getRotation, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(length, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T);

template <typename T>
MAKE_CUDA_KERNEL(isLeft, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(distanceTo, D, T);
#endif

template <Size D, typename T>
TEST_SUITE(LineSegment)
{
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(getRotation, 1, 1, T);
  TEST_HOSTDEV(length, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  TEST_HOSTDEV(isLeft, 1, 1, T);
  TEST_HOSTDEV(distanceTo, 1, 1, D, T);
}

auto
main() -> int
{
  RUN_SUITE((LineSegment<2, float>));
  RUN_SUITE((LineSegment<3, float>));
  RUN_SUITE((LineSegment<2, double>));
  RUN_SUITE((LineSegment<3, double>));
  return 0;
}
