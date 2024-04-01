#include <um2/geometry/triangle.hpp>
#include <um2/geometry/modular_rays.hpp>

#include "../../test_macros.hpp"

Float constexpr eps = um2::eps_distance;

template <Int D>
HOSTDEV constexpr auto
makeTri() -> um2::Triangle<D>
{
  um2::Triangle<D> this_tri;
  this_tri[0] = 0; 
  this_tri[1] = 0; 
  this_tri[2] = 0; 
  this_tri[1][0] = castIfNot<Float>(1);
  this_tri[2][1] = castIfNot<Float>(1);
  return this_tri;
}

//==============================================================================
// interpolation
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Triangle<D> tri = makeTri<D>();
  um2::Point<D> const p00 = tri(0, 0);
  um2::Point<D> const p10 = tri(1, 0);
  um2::Point<D> const p01 = tri(0, 1);
  ASSERT(p00.isApprox(tri[0]));
  ASSERT(p10.isApprox(tri[1]));
  ASSERT(p01.isApprox(tri[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::Triangle<D> tri = makeTri<D>();
  auto jac = tri.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
  jac = tri.jacobian(static_cast<Float>(0.2), static_cast<Float>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
  // If we stretch the triangle, the Jacobian should change.
  tri[1][0] = static_cast<Float>(2);
  jac = tri.jacobian(0.5, 0);
  ASSERT_NEAR((jac(0, 0)), 2, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(getEdge)
{
  um2::Triangle<D> tri = makeTri<D>();
  um2::LineSegment<D> edge = tri.getEdge(0);
  ASSERT(edge[0].isApprox(tri[0]));
  ASSERT(edge[1].isApprox(tri[1]));
  edge = tri.getEdge(1);
  ASSERT(edge[0].isApprox(tri[1]));
  ASSERT(edge[1].isApprox(tri[2]));
  edge = tri.getEdge(2);
  ASSERT(edge[0].isApprox(tri[2]));
  ASSERT(edge[1].isApprox(tri[0]));
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Triangle<D> const tri = makeTri<D>();
  auto const two = castIfNot<Float>(2);
  Float const ref = two + um2::sqrt(two);
  ASSERT_NEAR(tri.perimeter(), ref, eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::AxisAlignedBox<D> const box = tri.boundingBox();
  ASSERT_NEAR(box.minima()[0], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima()[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima()[1], castIfNot<Float>(1), eps);
}

//==============================================================================
// area
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(area)
{
  um2::Triangle<D> tri = makeTri<D>();
  ASSERT_NEAR(tri.area(), castIfNot<Float>(0.5), eps);
  tri[1][0] = castIfNot<Float>(2);
  ASSERT_NEAR(tri.area(), castIfNot<Float>(1), eps);
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::Point<D> const c = tri.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(1.0 / 3.0), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(1.0 / 3.0), eps);
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flip)
{
  um2::Triangle<2> tri = makeTri<2>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  ASSERT(!tri.isCCW());
  tri.flip();
  ASSERT(tri.isCCW());
}

//==============================================================================
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::Triangle<2> const tri = makeTri<2>();
  um2::Point2 p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(1.25), castIfNot<Float>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(-0.25));
  ASSERT(!tri.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  um2::Triangle<2> const tri = makeTri<2>();
  auto const two = castIfNot<Float>(2);
  auto const ref = um2::pi<Float> / (two * (two + um2::sqrt(two)));
  ASSERT_NEAR(tri.meanChordLength(), ref, eps);
}

//==============================================================================
// intersect
//=============================================================================

HOSTDEV
void
testTriForIntersections(um2::Triangle<2> const tri)
{
  // Parameters
  Int constexpr num_angles = 64; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto aabb = tri.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
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
      Float buf[3];
      auto const hits = tri.intersect(ray, buf);
      // For each intersection coordinate
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2 const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        Float min_dist = um2::inf_distance;
        for (Int ie = 0; ie < 3; ++ie) {
          um2::LineSegment<2> const l = tri.getEdge(ie);
          Float const d = l.distanceTo(p);
          if (d < min_dist) {
            min_dist = d;
          }
        }
        ASSERT(min_dist < um2::eps_distance);
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  um2::Triangle<2> tri = makeTri<2>();
  testTriForIntersections(tri);
  tri[1][0] = castIfNot<Float>(2);
  testTriForIntersections(tri);
}

template <Int D>
TEST_SUITE(Triangle)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(getEdge, D);
  TEST_HOSTDEV(perimeter, D);
  TEST_HOSTDEV(boundingBox, D);
  TEST_HOSTDEV(area, D);
  TEST_HOSTDEV(centroid, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isCCW_flip);
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(meanChordLength);
    TEST_HOSTDEV(intersect);
  }
}

auto
main() -> int
{
  RUN_SUITE(Triangle<2>);
  RUN_SUITE(Triangle<3>);
  return 0;
}
