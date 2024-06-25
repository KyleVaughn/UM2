#include <um2/config.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/stdlib/numbers.hpp>

// NOLINTNEXTLINE(misc-include-cleaner) false positive
#include <um2/geometry/line_segment.hpp>

#include "../../test_macros.hpp"

// CUDA is annoying and defines half, so we have to use ahalf
template <class T>
inline constexpr T eps = um2::epsDistance<T>();

template <class T>
inline constexpr T ahalf = 1 / static_cast<T>(2);

template <Int D, class T>
HOSTDEV constexpr auto
makeLine() -> um2::LineSegment<D, T>
{
  um2::LineSegment<D, T> line;
  line[0] = 1; // all 1's
  line[1] = 2; // all 2's
  return line;
}

//=============================================================================
// Accessors
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::Point<D, T> p0;
  p0 = 1; // all 1's
  um2::Point<D, T> p1;
  p1 = 2; // all 2's
  um2::LineSegment<D, T> line(p0, p1);
  ASSERT(line[0].isApprox(p0));
  ASSERT(line[1].isApprox(p1));
  ASSERT(line.vertices()[0].isApprox(p0));
  ASSERT(line.vertices()[1].isApprox(p1));
  line[0] = p1;
  ASSERT(line[0].isApprox(p1));
}

//=============================================================================
// Interpolation
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Point<D, T> const p0 = line(0);
  ASSERT(p0.isApprox(line[0]));
  um2::Point<D, T> const p1 = line(1);
  ASSERT(p1.isApprox(line[1]));
  um2::Point<D, T> const p05 = line(ahalf<T>);
  um2::Point<D, T> const mp = um2::midpoint(p0, p1);
  ASSERT(p05.isApprox(mp));
}

//=============================================================================
// jacobian
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D, T> const line = makeLine<D, T>();
  auto const j = line.jacobian(0);
  um2::Point<D, T> ref;
  ref = 1; // all 1's
  ASSERT(j.isApprox(ref));
}

//=============================================================================
// length
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D, T> const line = makeLine<D, T>();
  T const len_ref = um2::sqrt(static_cast<T>(D));
  ASSERT_NEAR(line.length(), len_ref, eps<T>);
}
//=============================================================================
// boundingBox
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  auto const box = line.boundingBox();
  ASSERT(line[0].isApprox(box.minima()));
  ASSERT(line[1].isApprox(box.maxima()));
}

//=============================================================================
// pointClosestTo
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(pointClosestTo)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();

  // The left end point
  um2::Point<D, T> p0 = line[0];
  ASSERT_NEAR(line.pointClosestTo(p0), 0, eps<T>);
  // A point to the left of the left end point
  p0[0] -= 1;
  ASSERT_NEAR(line.pointClosestTo(p0), 0, eps<T>);
  // A point to the right of the left end point
  p0 = line(ahalf<T>);
  p0[0] -= static_cast<T>(1) / 10;
  p0[1] += static_cast<T>(1) / 10;
  ASSERT_NEAR(line.pointClosestTo(p0), ahalf<T>, eps<T>);

  // Repeat for the right end point
  um2::Point<D, T> p1 = line[1];
  ASSERT_NEAR(line.pointClosestTo(p1), 1, eps<T>);
  p1[0] += 1;
  ASSERT_NEAR(line.pointClosestTo(p1), 1, eps<T>);
}

//=============================================================================
// distanceTo
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(distanceTo)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();

  // The left end point
  um2::Point<D, T> p0 = line[0];
  ASSERT_NEAR(line.distanceTo(p0), 0, eps<T>);
  // A point to the left of the left end point
  p0[0] -= 1;
  ASSERT_NEAR(line.distanceTo(p0), 1, eps<T>);
  // A point to the right of the left end point
  p0[0] += ahalf<T> * 3;
  T ref = 0;
  if constexpr (D == 2) {
    ref = um2::sin(um2::pi_4<T>) / 2;
  } else {
    // d = (7/6, 1/6, 1/6)
    ref = um2::sqrt(static_cast<T>(6)) / 6;
  }
  ASSERT_NEAR(line.distanceTo(p0), ref, eps<T>);

  // Repeat for the right end point
  um2::Point<D, T> p1 = line[1];
  ASSERT_NEAR(line.distanceTo(p1), 0, eps<T>);
  p1[0] += 1;
  ASSERT_NEAR(line.distanceTo(p1), 1, eps<T>);
  p1[0] -= ahalf<T> * 3;
  ASSERT_NEAR(line.distanceTo(p1), ref, eps<T>);
}

//=============================================================================
// getRotation
//=============================================================================

template <class T>
HOSTDEV
TEST_CASE(getRotation)
{
  // Anchor p0 at (0, 0) and rotate p1 around a circle
  um2::Point2<T> const p0(0, 0);
  T const dang = um2::pi<T> / 128;
  T ang = dang;
  while (ang < 2 * um2::pi<T>) {
    um2::Point2<T> const p1(um2::cos(ang), um2::sin(ang));
    um2::LineSegment2<T> const line(p0, p1);
    auto const r = line.getRotation();
    auto const p1_rot = r * p1;
    ASSERT_NEAR(p1_rot[0], 1, eps<T>);
    ASSERT_NEAR(p1_rot[1], 0, eps<T>);
    ang += dang;
  }
}

//=============================================================================
// isLeft
//=============================================================================

template <class T>
HOSTDEV
TEST_CASE(isLeft)
{
  um2::LineSegment2<T> line = makeLine<2, T>();
  um2::Point2<T> p0 = line[0];
  um2::Point2<T> p1 = line[1];
  p0[1] -= 1; // (1, 0)
  p1[1] += 1; // (2, 3)
  ASSERT(!line.isLeft(p0));
  ASSERT(line.isLeft(p1));
  p0[1] += static_cast<T>(2); // (1, 2)
  p1[1] -= static_cast<T>(2); // (2, 1)
  ASSERT(line.isLeft(p0));
  ASSERT(!line.isLeft(p1));
  p1[1] = static_cast<T>(2) + static_cast<T>(1) / 100; // (2, 2.01)
  ASSERT(line.isLeft(p1));
}

//=============================================================================
// intersect
//=============================================================================

template <class T>
HOSTDEV void
testEdgeForIntersections(um2::LineSegment2<T> const & l)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 100;

  auto aabb = l.boundingBox();
  aabb.scale(castIfNot<T>(1.1));
  auto const longest_edge =
      aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<T>(rays_per_longest_edge);
  T const pi_deg = um2::pi_2<T> / static_cast<T>(num_angles);
  T r = 0;
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    T const angle = pi_deg * static_cast<T>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto const num_valid = l.intersect(ray, &r);
      if (0 < num_valid) {
        um2::Point2<T> const p = ray(r);
        T const d = l.distanceTo(p);
        ASSERT(d < um2::epsDistance<T>());
      }
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(intersect)
{
  um2::LineSegment2<T> l(um2::Point2<T>(0, 1), um2::Point2<T>(2, -1));
  um2::Ray2<T> const ray(um2::Point2<T>(0, -1), um2::normalized(um2::Point2<T>(1, 1)));
  T res = 0;
  Int num_valid = l.intersect(ray, &res);
  ASSERT(num_valid == 1);
  ASSERT_NEAR(res, um2::sqrt(static_cast<T>(2)), eps<T>);
  res = 0;

  l = um2::LineSegment2<T>(um2::Point2<T>(1, -1), um2::Point2<T>(1, 1));
  num_valid = l.intersect(ray, &res);
  ASSERT(num_valid == 1);
  ASSERT_NEAR(res, um2::sqrt(static_cast<T>(2)), eps<T>);

  // Anchor p0 at (0, 0) and rotate p1 around a circle
  um2::Point2<T> const p0(0, 0);
  T const dang = um2::pi<T> / 128;
  T ang = dang;
  while (ang < 2 * um2::pi<T>) {
    um2::Point2<T> const p1(um2::cos(ang), um2::sin(ang));
    um2::LineSegment2<T> const line(p0, p1);
    testEdgeForIntersections(line);
    ang += dang;
  }
}

#if UM2_USE_CUDA
template <Int D, class T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(interpolate, D, T);

#endif

template <Int D, class T>
TEST_SUITE(LineSegment)
{

  TEST_HOSTDEV(accessors, D, T);
  TEST_HOSTDEV(interpolate, D, T);
  TEST_HOSTDEV(jacobian, D, T);
  TEST_HOSTDEV(length, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  TEST_HOSTDEV(pointClosestTo, D, T);
  TEST_HOSTDEV(distanceTo, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(getRotation, T);
    TEST_HOSTDEV(isLeft, T);
    TEST_HOSTDEV(intersect, T);
  }
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
