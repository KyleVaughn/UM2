#include <um2/geometry/quadrilateral.hpp>
#include <um2/geometry/modular_rays.hpp>

#include "../../test_macros.hpp"

Float constexpr eps = um2::eps_distance;

template <Int D>
HOSTDEV constexpr auto
makeQuad() -> um2::Quadrilateral<D>
{
  um2::Quadrilateral<D> quad;
  for (Int i = 0; i < 4; ++i) {
    quad[i]= 0;
  }
  quad[1][0] = castIfNot<Float>(1);
  quad[2][0] = castIfNot<Float>(1);
  quad[2][1] = castIfNot<Float>(1);
  quad[3][1] = castIfNot<Float>(1);
  return quad;
}

template <Int D>
HOSTDEV constexpr auto
makeTriQuad() -> um2::Quadrilateral<D>
{
  um2::Quadrilateral<D> quad;
  for (Int i = 0; i < 4; ++i) {
    quad[i] = 0; 
  }
  quad[1][0] = castIfNot<Float>(1);
  quad[2][0] = castIfNot<Float>(1);
  quad[2][1] = castIfNot<Float>(1);
  quad[3][1] = castIfNot<Float>(0.5);
  quad[3][0] = castIfNot<Float>(0.5);
  return quad;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Quadrilateral<D> quad = makeQuad<D>();
  um2::Point<D> const p00 = quad(0, 0);
  um2::Point<D> const p10 = quad(1, 0);
  um2::Point<D> const p01 = quad(0, 1);
  um2::Point<D> const p11 = quad(1, 1);
  ASSERT(p00.isApprox(quad[0]));
  ASSERT(p10.isApprox(quad[1]));
  ASSERT(p01.isApprox(quad[3]));
  ASSERT(p11.isApprox(quad[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D> 
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::Quadrilateral<D> quad = makeQuad<D>();
  auto jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);

  jac = quad.jacobian(static_cast<Float>(0.2), static_cast<Float>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);

  // Extend in x-direction.
  quad[1][0] = castIfNot<Float>(2);
  quad[2][0] = castIfNot<Float>(2);
  jac = quad.jacobian(0, 0);
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
  um2::Quadrilateral<D> quad = makeQuad<D>();
  um2::LineSegment<D> edge = quad.getEdge(0);
  ASSERT(edge[0].isApprox(quad[0]));
  ASSERT(edge[1].isApprox(quad[1]));
  edge = quad.getEdge(1);
  ASSERT(edge[0].isApprox(quad[1]));
  ASSERT(edge[1].isApprox(quad[2]));
  edge = quad.getEdge(2);
  ASSERT(edge[0].isApprox(quad[2]));
  ASSERT(edge[1].isApprox(quad[3]));
  edge = quad.getEdge(3);
  ASSERT(edge[0].isApprox(quad[3]));
  ASSERT(edge[1].isApprox(quad[0]));
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Quadrilateral<D> const quad = makeQuad<D>();
  ASSERT_NEAR(quad.perimeter(), castIfNot<Float>(4), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Quadrilateral<D> const quad = makeQuad<D>();
  um2::AxisAlignedBox<D> const box = quad.boundingBox();
  ASSERT_NEAR(box.minima()[0], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima()[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima()[1], castIfNot<Float>(1), eps);
}

//==============================================================================
// isConvex
//==============================================================================

HOSTDEV
TEST_CASE(isConvex)
{
  um2::Quadrilateral<2> quad = makeQuad<2>();
  ASSERT(quad.isConvex());
  quad[3][0] = castIfNot<Float>(0.5);
  ASSERT(quad.isConvex());
  quad[3][1] = castIfNot<Float>(0.5);
  ASSERT(quad.isConvex()); // Effectively a triangle.
  quad[3][0] = castIfNot<Float>(0.75);
  ASSERT(!quad.isConvex());
}

//==============================================================================
// area
//==============================================================================

HOSTDEV
TEST_CASE(area)
{
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  // Compiler has issues if we make this a static_assert.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(quad.area(), castIfNot<Float>(1), eps);
  um2::Quadrilateral<2> const triquad = makeTriQuad<2>();
  ASSERT_NEAR(triquad.area(), castIfNot<Float>(0.5), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// centroid
//==============================================================================

HOSTDEV
TEST_CASE(centroid)
{
  um2::Quadrilateral<2> quad = makeQuad<2>();
  um2::Point<2> c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(0.5), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(0.5), eps);
  quad[2] = um2::Point<2>(castIfNot<Float>(2), castIfNot<Float>(0.5));
  quad[3] = um2::Point<2>(castIfNot<Float>(1), castIfNot<Float>(0.5));
  c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(1.00), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(0.25), eps);
  um2::Quadrilateral<2> const quad2 = makeTriQuad<2>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(castIfNot<Float>(2) / 3), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(castIfNot<Float>(1) / 3), eps);
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flip)
{
  um2::Quadrilateral<2> quad = makeQuad<2>();
  ASSERT(quad.isCCW());
  um2::swap(quad[1], quad[3]);
  ASSERT(!quad.isCCW());
  quad.flip();
  ASSERT(quad.isCCW());
}

//==============================================================================
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  um2::Point2 p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(castIfNot<Float>(1.25), castIfNot<Float>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(-0.25));
  ASSERT(!quad.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  ASSERT_NEAR(quad.meanChordLength(), um2::pi_4<Float>, eps);
}

//==============================================================================
// intersect
//=============================================================================

HOSTDEV
void
testQuadForIntersections(um2::Quadrilateral2 const & quad)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto const aabb = quad.boundingBox();
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
      Float buf[4];
      auto const hits = quad.intersect(ray, buf);
      // For each intersection coordinate
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2 const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        Float min_dist = um2::inf_distance;
        for (Int ie = 0; ie < 4; ++ie) {
          um2::LineSegment<2> const l = quad.getEdge(ie);
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
  um2::Quadrilateral2 quad = makeQuad<2>();
  testQuadForIntersections(quad);
  quad = makeTriQuad<2>();
  testQuadForIntersections(quad);
}

template <Int D>
TEST_SUITE(Quadrilateral)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(getEdge, D);
  TEST_HOSTDEV(perimeter, D);
  TEST_HOSTDEV(boundingBox, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isConvex);
    TEST_HOSTDEV(area);
    TEST_HOSTDEV(centroid);
    TEST_HOSTDEV(isCCW_flip);
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(meanChordLength);
    TEST_HOSTDEV(intersect);
  }
}

auto
main() -> int
{
//  RUN_SUITE(Quadrilateral<2>);
  RUN_SUITE(Quadrilateral<3>);
  return 0;
}
