#include <um2/geometry/dion.hpp>
#include <um2/stdlib/numbers.hpp>

#include "../../test_macros.hpp"

// CUDA is annoying and defines half, so we have to use ahalf
Float constexpr eps = um2::eps_distance * static_cast<Float>(10);
Float constexpr ahalf = static_cast<Float>(1) / static_cast<Float>(2);

template <Int D>
HOSTDEV constexpr auto
makeLine() -> um2::LineSegment<D>
{
  um2::LineSegment<D> line;
  for (Int i = 0; i < D; ++i) {
    line[0][i] = static_cast<Float>(1);
    line[1][i] = static_cast<Float>(2);
  }
  return line;
}

//=============================================================================
// Intnterpolation
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::Point<D> const p0 = line(0);
  ASSERT((um2::isApprox<D>(p0, line[0])));
  um2::Point<D> const p1 = line(1);
  ASSERT((um2::isApprox<D>(p1, line[1])));
  um2::Point<D> const p05 = um2::interpolate(line, ahalf);
  um2::Point<D> const mp = um2::midpoint(p0, p1);
  ASSERT((um2::isApprox(p05, mp)));
}

//=============================================================================
// jacobian
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::Vec<D, Float> j_ref;
  for (Int i = 0; i < D; ++i) {
    j_ref[i] = line[1][i] - line[0][i];
  }
  um2::Vec<D, Float> const j0 = line.jacobian(0);
  um2::Vec<D, Float> const j1 = um2::jacobian(line, 1);
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
  um2::Mat2x2<Float> rot = line.getRotation();
  um2::LineSegment<2> line_rot(rot * line[0], rot * line[1]);
  ASSERT_NEAR(line_rot[0][1], static_cast<Float>(0), eps);
  ASSERT_NEAR(line_rot[1][1], static_cast<Float>(0), eps);
  um2::LineSegment<2> line_rot2(um2::Vec2<Float>::zero(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot2[0][0], static_cast<Float>(0), eps);
  ASSERT_NEAR(line_rot2[0][1], static_cast<Float>(0), eps);
  ASSERT_NEAR(line_rot2[1][1], static_cast<Float>(0), eps);
  line[0][0] = static_cast<Float>(10);
  rot = line.getRotation();
  um2::LineSegment<2> line_rot3(um2::Vec2<Float>::zero(), rot * (line[1] - line[0]));
  ASSERT_NEAR(line_rot3[0][0], static_cast<Float>(0), eps);
  ASSERT_NEAR(line_rot3[0][1], static_cast<Float>(0), eps);
  ASSERT_NEAR(line_rot3[1][1], static_cast<Float>(0), eps);
}

//=============================================================================
// length
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D> line = makeLine<D>();
  Float const len_ref = line[0].distanceTo(line[1]);
  ASSERT_NEAR(line.length(), len_ref, eps);
  ASSERT_NEAR(um2::length(line), len_ref, eps);
}

//=============================================================================
// boundingBox
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::AxisAlignedBox<D> const box = line.boundingBox();
  ASSERT(um2::isApprox<D>(line[0], box.minima()));
  ASSERT(um2::isApprox<D>(line[1], box.maxima()));
  um2::AxisAlignedBox<D> const box2 = um2::boundingBox(line);
  ASSERT(um2::isApprox<D>(box2.minima(), box.minima()));
  ASSERT(um2::isApprox<D>(box2.maxima(), box.maxima()));
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
  p0[1] -= static_cast<Float>(1); // (1, 0)
  p1[1] += static_cast<Float>(1); // (2, 3)
  ASSERT(!line.isLeft(p0));
  ASSERT(line.isLeft(p1));
  p0[1] += static_cast<Float>(2); // (1, 2)
  p1[1] -= static_cast<Float>(2); // (2, 1)
  ASSERT(line.isLeft(p0));
  ASSERT(!line.isLeft(p1));
  p1[1] = static_cast<Float>(2) + static_cast<Float>(1) / 100; // (2, 2.01)
  ASSERT(line.isLeft(p1));
}

//=============================================================================
// pointClosestTo
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(pointClosestTo)
{
  um2::LineSegment<D> line = makeLine<D>();

  // The left end point
  um2::Point<D> p0 = line[0];
  ASSERT_NEAR(line.pointClosestTo(p0), static_cast<Float>(0), eps);
  // A point to the left of the left end point
  p0[0] -= static_cast<Float>(1);
  ASSERT_NEAR(line.pointClosestTo(p0), static_cast<Float>(0), eps);
  // A point to the right of the left end point
  p0 = line(ahalf);
  p0[0] -= static_cast<Float>(1) / 10;
  p0[1] += static_cast<Float>(1) / 10;
  ASSERT_NEAR(line.pointClosestTo(p0), ahalf, eps);

  // Repeat for the right end point
  um2::Point<D> p1 = line[1];
  ASSERT_NEAR(line.pointClosestTo(p1), static_cast<Float>(1), eps);
  p1[0] += static_cast<Float>(1);
  ASSERT_NEAR(line.pointClosestTo(p1), static_cast<Float>(1), eps);
}

//=============================================================================
// distanceTo
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(distanceTo)
{
  um2::LineSegment<D> line = makeLine<D>();

  // The left end point
  um2::Point<D> p0 = line[0];
  ASSERT_NEAR(line.distanceTo(p0), static_cast<Float>(0), eps);
  // A point to the left of the left end point
  p0[0] -= static_cast<Float>(1);
  ASSERT_NEAR(line.distanceTo(p0), static_cast<Float>(1), eps);
  // A point to the right of the left end point
  p0[0] += ahalf * 3;
  Float ref = 0;
  if constexpr (D == 2) {
    ref = um2::sin(um2::pi_4<Float>) / 2;
  } else {
    // d = (7/6, 1/6, 1/6)
    ref = um2::sqrt(static_cast<Float>(6)) / 6;
  }
  ASSERT_NEAR(line.distanceTo(p0), ref, eps);

  // Repeat for the right end point
  um2::Point<D> p1 = line[1];
  ASSERT_NEAR(line.distanceTo(p1), static_cast<Float>(0), eps);
  p1[0] += static_cast<Float>(1);
  ASSERT_NEAR(line.distanceTo(p1), static_cast<Float>(1), eps);
  p1[0] -= ahalf * 3;
  ASSERT_NEAR(line.distanceTo(p1), ref, eps);
}

HOSTDEV
TEST_CASE(intersect)
{
  um2::LineSegment2 l(um2::Point2(0, 1), um2::Point2(2, -1));
  um2::Ray2 const ray(um2::Point2(0, -1), um2::normalized(um2::Point2(1, 1)));
  Float res = um2::intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<Float>(2)), eps * 100);

  l = um2::LineSegment2(um2::Point2(1, -1), um2::Point2(1, 1));
  res = um2::intersect(ray, l);
  ASSERT_NEAR(res, um2::sqrt(static_cast<Float>(2)), eps * 100);
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Int D>
MAKE_CUDA_KERNEL(jacobian, D);

MAKE_CUDA_KERNEL(getRotation);

template <Int D>
MAKE_CUDA_KERNEL(length, D);

template <Int D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isLeft);

template <Int D>
MAKE_CUDA_KERNEL(distanceTo, D);

template <Int D>
MAKE_CUDA_KERNEL(pointClosestTo, D);

MAKE_CUDA_KERNEL(intersect);
#endif

template <Int D>
TEST_SUITE(LineSegment)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(length, D);
  TEST_HOSTDEV(boundingBox, D);
  TEST_HOSTDEV(pointClosestTo, D);
  TEST_HOSTDEV(distanceTo, D);
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
