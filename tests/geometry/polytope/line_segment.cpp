#include <um2/geometry/line_segment.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/geometry/modular_rays.hpp>    
    
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
// Interpolation
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D> line = makeLine<D>();
  um2::Point<D> const p0 = line(0);
  ASSERT(p0.isApprox(line[0]));
  um2::Point<D> const p1 = line(1);
  ASSERT(p1.isApprox(line[1]));
  um2::Point<D> const p05 = line(ahalf);
  um2::Point<D> const mp = um2::midpoint(p0, p1);
  ASSERT(p05.isApprox(mp));
}

//=============================================================================
// getRotation
//=============================================================================

HOSTDEV
TEST_CASE(getRotation)
{
  // Anchor p0 at (0, 0) and rotate p1 around a circle    
  um2::Point2 const p0(0, 0);    
  Float const dang = um2::pi<Float> / 128;    
  Float ang = dang;    
  while (ang < 2 * um2::pi<Float>) {    
    um2::Point2 const p1(um2::cos(ang), um2::sin(ang));    
    um2::LineSegment2 const line(p0, p1);    
    auto const r = line.getRotation();
    auto const p1_rot = r * p1;
    ASSERT_NEAR(p1_rot[0], 1, eps);
    ASSERT_NEAR(p1_rot[1], 0, eps);
    ang += dang;    
  } 
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
  ASSERT(line[0].isApprox(box.minima()));
  ASSERT(line[1].isApprox(box.maxima()));
  um2::AxisAlignedBox<D> const box2 = um2::boundingBox(line);
  ASSERT(box2.minima().isApprox(box.minima()));
  ASSERT(box2.maxima().isApprox(box.maxima()));
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
  ASSERT_NEAR(line.pointClosestTo(p1), 1, eps);
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
void
testEdgeForIntersections(um2::LineSegment2 const & l)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 100;

  auto aabb = l.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto const r = l.intersect(ray);
      if (r < um2::inf_distance / 10) {
        um2::Point2 const p = ray(r);
        Float const d = l.distanceTo(p);
        ASSERT(d < 10 * um2::eps_distance);
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  um2::LineSegment2 l(um2::Point2(0, 1), um2::Point2(2, -1));
  um2::Ray2 const ray(um2::Point2(0, -1), um2::normalized(um2::Point2(1, 1)));
  Float res = l.intersect(ray);
  ASSERT_NEAR(res, um2::sqrt(static_cast<Float>(2)), eps * 100);

  l = um2::LineSegment2(um2::Point2(1, -1), um2::Point2(1, 1));
  res = l.intersect(ray);
  ASSERT_NEAR(res, um2::sqrt(static_cast<Float>(2)), eps * 100);

  // Anchor p0 at (0, 0) and rotate p1 around a circle
  um2::Point2 const p0(0, 0);
  Float const dang = um2::pi<Float> / 128;
  Float ang = dang;
  while (ang < 2 * um2::pi<Float>) {
    um2::Point2 const p1(um2::cos(ang), um2::sin(ang));
    um2::LineSegment2 const line(p0, p1);
    testEdgeForIntersections(line);
    ang += dang;
  }
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

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
