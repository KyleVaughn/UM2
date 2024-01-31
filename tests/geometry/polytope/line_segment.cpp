#include <um2/geometry/dion.hpp>

#include "../../test_macros.hpp"

F constexpr eps = um2::eps_distance * static_cast<F>(10);
F constexpr half = static_cast<F>(1) / static_cast<F>(2);

template <I D>
HOSTDEV constexpr auto
makeLine() -> um2::LineSegment<D>
{
  um2::LineSegment<D> line;
  for (I i = 0; i < D; ++i) {
    line[0][i] = static_cast<F>(1);
    line[1][i] = static_cast<F>(2);
  }
  return line;
}

//=============================================================================
// Interpolation
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::Point<D> const p0 = line(0);
  ASSERT((um2::isApprox(p0, line[0])));
  um2::Point<D> const p1 = line(1);
  ASSERT((um2::isApprox(p1, line[1])));
  um2::Point<D> const p05 = um2::interpolate(line, half);
  um2::Point<D> const mp = um2::midpoint(p0, p1);
  ASSERT((um2::isApprox(p05, mp)));
}

//=============================================================================
// jacobian
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::Vec<D, F> j_ref;
  for (I i = 0; i < D; ++i) {
    j_ref[i] = line[1][i] - line[0][i];
  }
  um2::Vec<D, F> const j0 = line.jacobian(0);
  um2::Vec<D, F> const j1 = um2::jacobian(line, 1);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));
}

//=============================================================================
// getRotation
//=============================================================================

HOSTDEV
TEST_CASE(getRotation)
{
  um2::LineSegment<2> line = makeLine<2>();
  um2::Mat2x2<F> rot = line.getRotation();
  um2::LineSegment<2> line_rot(rot * line[0], rot * line[1]);
  ASSERT_NEAR(line_rot[0][1], static_cast<F>(0), eps);
  ASSERT_NEAR(line_rot[1][1], static_cast<F>(0), eps);
  um2::LineSegment<2> line_rot2(um2::Vec2<F>::zero(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot2[0][0], static_cast<F>(0), eps);
  ASSERT_NEAR(line_rot2[0][1], static_cast<F>(0), eps);
  ASSERT_NEAR(line_rot2[1][1], static_cast<F>(0), eps);
  line[0][0] = static_cast<F>(10);
  rot = line.getRotation();
  um2::LineSegment<2> line_rot3(um2::Vec2<F>::zero(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot3[0][0], static_cast<F>(0), eps);
  ASSERT_NEAR(line_rot3[0][1], static_cast<F>(0), eps);
  ASSERT_NEAR(line_rot3[1][1], static_cast<F>(0), eps);
}

//=============================================================================
// length
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D> line = makeLine<D>();
  F const len_ref = line[0].distanceTo(line[1]);
  ASSERT_NEAR(line.length(), len_ref, eps);
  ASSERT_NEAR(um2::length(line), len_ref, eps);
}

//=============================================================================
// boundingBox
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::AxisAlignedBox<D> const box = line.boundingBox();
  ASSERT(um2::isApprox(line[0], box.minima()));
  ASSERT(um2::isApprox(line[1], box.maxima()));
  um2::AxisAlignedBox<D> const box2 = um2::boundingBox(line);
  ASSERT(um2::isApprox(box2.minima(), box.minima()));
  ASSERT(um2::isApprox(box2.maxima(), box.maxima()));
}

//=============================================================================
// isLeft
//=============================================================================

HOSTDEV
TEST_CASE(isLeft)
{
  um2::LineSegment2 line = makeLine<2>();
  um2::Point2 p0 = line[0];
  um2::Point2 p1 = line[1];
  p0[1] -= static_cast<F>(1); // (1, 0)
  p1[1] += static_cast<F>(1); // (2, 3)
  ASSERT(!line.isLeft(p0));
  ASSERT(line.isLeft(p1));
  p0[1] += static_cast<F>(2); // (1, 2)
  p1[1] -= static_cast<F>(2); // (2, 1)
  ASSERT(line.isLeft(p0));
  ASSERT(!line.isLeft(p1));
  p1[1] = static_cast<F>(2) + static_cast<F>(1) / 100; // (2, 2.01)
  ASSERT(line.isLeft(p1));
}

//=============================================================================
// pointClosestTo
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(pointClosestTo)
{
  um2::LineSegment<D> line = makeLine<D>();

  // The left end point
  um2::Point<D> p0 = line[0];
  ASSERT_NEAR(line.pointClosestTo(p0), static_cast<F>(0), eps);
  // A point to the left of the left end point
  p0[0] -= static_cast<F>(1);
  ASSERT_NEAR(line.pointClosestTo(p0), static_cast<F>(0), eps);
  // A point to the right of the left end point
  p0 = line(half);
  p0[0] -= static_cast<F>(1) / 10;
  p0[1] += static_cast<F>(1) / 10;
  ASSERT_NEAR(line.pointClosestTo(p0), half, eps);

  // Repeat for the right end point
  um2::Point<D> p1 = line[1];
  ASSERT_NEAR(line.pointClosestTo(p1), static_cast<F>(1), eps);
  p1[0] += static_cast<F>(1);
  ASSERT_NEAR(line.pointClosestTo(p1), static_cast<F>(1), eps);
}

//=============================================================================
// distanceTo
//=============================================================================

template <I D>
HOSTDEV
TEST_CASE(distanceTo)
{
  um2::LineSegment<D> line = makeLine<D>();

  // The left end point
  um2::Point<D> p0 = line[0];
  ASSERT_NEAR(line.distanceTo(p0), static_cast<F>(0), eps);
  // A point to the left of the left end point
  p0[0] -= static_cast<F>(1);
  ASSERT_NEAR(line.distanceTo(p0), static_cast<F>(1), eps);
  // A point to the right of the left end point
  p0[0] += half * 3;
  F ref = 0;
  if constexpr (D == 2) {
    ref = um2::sin(um2::pi<F> / static_cast<F>(4)) / 2;
  } else {
    // d = (7/6, 1/6, 1/6)
    ref = um2::sqrt(static_cast<F>(6)) / 6;
  }
  ASSERT_NEAR(line.distanceTo(p0), ref, eps);

  // Repeat for the right end point
  um2::Point<D> p1 = line[1];
  ASSERT_NEAR(line.distanceTo(p1), static_cast<F>(0), eps);
  p1[0] += static_cast<F>(1);
  ASSERT_NEAR(line.distanceTo(p1), static_cast<F>(1), eps);
  p1[0] -= half * 3;
  ASSERT_NEAR(line.distanceTo(p1), ref, eps);
}

HOSTDEV
TEST_CASE(intersect)
{
  um2::LineSegment2 l(um2::Point2(0, 1), um2::Point2(2, -1));
  um2::Ray2 const ray(um2::Point2(0, -1), um2::normalized(um2::Point2(1, 1)));
  F res = um2::intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<F>(2)), eps * 100);

  l = um2::LineSegment2(um2::Point2(1, -1), um2::Point2(1, 1));
  res = um2::intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<F>(2)), eps * 100);
}

#if UM2_USE_CUDA
template <I D>
MAKE_CUDA_KERNEL(interpolate, D);

template <I D>
MAKE_CUDA_KERNEL(jacobian, D);

MAKE_CUDA_KERNEL(getRotation);

template <I D>
MAKE_CUDA_KERNEL(length, D);

template <I D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isLeft);

template <I D>
MAKE_CUDA_KERNEL(distanceTo, D);

template <I D>
MAKE_CUDA_KERNEL(pointClosestTo, D);

MAKE_CUDA_KERNEL(intersect);
#endif

template <I D>
TEST_SUITE(LineSegment)
{
  TEST_HOSTDEV(interpolate, 1, 1, D);
  TEST_HOSTDEV(jacobian, 1, 1, D);
  TEST_HOSTDEV(length, 1, 1, D);
  TEST_HOSTDEV(boundingBox, 1, 1, D);
  TEST_HOSTDEV(pointClosestTo, 1, 1, D);
  TEST_HOSTDEV(distanceTo, 1, 1, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getRotation);
    TEST_HOSTDEV(isLeft);
    TEST_HOSTDEV(intersect);
  }
}

auto
main() -> int
{
  RUN_SUITE(LineSegment<2>);
  RUN_SUITE(LineSegment<3>);
  return 0;
}
