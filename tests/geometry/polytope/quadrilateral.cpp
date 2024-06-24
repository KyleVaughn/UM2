#include <um2/config.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/axis_aligned_box.hpp>

// NOLINTNEXTLINE(misc-include-cleaner)
#include <um2/geometry/quadrilateral.hpp>

#include "../../test_macros.hpp"

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeQuad() -> um2::Quadrilateral<D, T>
{
  um2::Quadrilateral<D, T> quad;
  for (Int i = 0; i < 4; ++i) {
    quad[i] = 0;
  }
  quad[1][0] = castIfNot<T>(1);
  quad[2][0] = castIfNot<T>(1);
  quad[2][1] = castIfNot<T>(1);
  quad[3][1] = castIfNot<T>(1);
  return quad;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeTriQuad() -> um2::Quadrilateral<D, T>
{
  um2::Quadrilateral<D, T> quad;
  for (Int i = 0; i < 4; ++i) {
    quad[i] = 0;
  }
  quad[1][0] = castIfNot<T>(1);
  quad[2][0] = castIfNot<T>(1);
  quad[2][1] = castIfNot<T>(1);
  quad[3][1] = castIfNot<T>(0.5);
  quad[3][0] = castIfNot<T>(0.5);
  return quad;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  um2::Point<D, T> const p00 = quad(0, 0);
  um2::Point<D, T> const p10 = quad(1, 0);
  um2::Point<D, T> const p01 = quad(0, 1);
  um2::Point<D, T> const p11 = quad(1, 1);
  ASSERT(p00.isApprox(quad[0]));
  ASSERT(p10.isApprox(quad[1]));
  ASSERT(p01.isApprox(quad[3]));
  ASSERT(p11.isApprox(quad[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  auto jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps<T>);
  ASSERT_NEAR((jac(1, 0)), 0, eps<T>);
  ASSERT_NEAR((jac(0, 1)), 0, eps<T>);
  ASSERT_NEAR((jac(1, 1)), 1, eps<T>);

  jac = quad.jacobian(castIfNot<T>(0.2), castIfNot<T>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps<T>);
  ASSERT_NEAR((jac(1, 0)), 0, eps<T>);
  ASSERT_NEAR((jac(0, 1)), 0, eps<T>);
  ASSERT_NEAR((jac(1, 1)), 1, eps<T>);

  // Extend in x-direction.
  quad[1][0] = castIfNot<T>(2);
  quad[2][0] = castIfNot<T>(2);
  jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 2, eps<T>);
  ASSERT_NEAR((jac(1, 0)), 0, eps<T>);
  ASSERT_NEAR((jac(0, 1)), 0, eps<T>);
  ASSERT_NEAR((jac(1, 1)), 1, eps<T>);
}

//==============================================================================
// getEdge
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(getEdge)
{
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  um2::LineSegment<D, T> edge = quad.getEdge(0);
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

template <Int D, class T>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Quadrilateral<D, T> const quad = makeQuad<D, T>();
  ASSERT_NEAR(quad.perimeter(), castIfNot<T>(4), eps<T>);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Quadrilateral<D, T> const quad = makeQuad<D, T>();
  um2::AxisAlignedBox<D, T> const box = quad.boundingBox();
  ASSERT_NEAR(box.minima()[0], castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.minima()[1], castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima()[0], castIfNot<T>(1), eps<T>);
  ASSERT_NEAR(box.maxima()[1], castIfNot<T>(1), eps<T>);
}

//==============================================================================
// isConvex
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(isConvex)
{
  um2::Quadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT(quad.isConvex());
  quad[3][0] = castIfNot<T>(0.5);
  ASSERT(quad.isConvex());
  quad[3][1] = castIfNot<T>(0.5);
  ASSERT(quad.isConvex()); // Effectively a triangle.
  quad[3][0] = castIfNot<T>(0.75);
  ASSERT(!quad.isConvex());
}

//==============================================================================
// area
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(area)
{
  um2::Quadrilateral<2, T> const quad = makeQuad<2, T>();
  // Compiler has issues if we make this a static_assert.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(quad.area(), castIfNot<T>(1), eps<T>);
  um2::Quadrilateral<2, T> const triquad = makeTriQuad<2, T>();
  ASSERT_NEAR(triquad.area(), castIfNot<T>(0.5), eps<T>);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// centroid
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Quadrilateral<2, T> quad = makeQuad<2, T>();
  um2::Point<2, T> c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(0.5), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(0.5), eps<T>);
  quad[2] = um2::Point<2, T>(castIfNot<T>(2), castIfNot<T>(0.5));
  quad[3] = um2::Point<2, T>(castIfNot<T>(1), castIfNot<T>(0.5));
  c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(1.00), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(0.25), eps<T>);
  um2::Quadrilateral<2, T> const quad2 = makeTriQuad<2, T>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(castIfNot<T>(2) / 3), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(castIfNot<T>(1) / 3), eps<T>);
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(isCCW_flip)
{
  um2::Quadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT(quad.isCCW());
  um2::swap(quad[1], quad[3]);
  ASSERT(!quad.isCCW());
  quad.flip();
  ASSERT(quad.isCCW());
}

//==============================================================================
// contains
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(contains)
{
  um2::Quadrilateral<2, T> const quad = makeQuad<2, T>();
  um2::Point2<T> p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.5), castIfNot<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(1.25), castIfNot<T>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(-0.25));
  ASSERT(!quad.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(meanChordLength)
{
  um2::Quadrilateral<2, T> const quad = makeQuad<2, T>();
  ASSERT_NEAR(quad.meanChordLength(), um2::pi_4<T>, eps<T>);
}

//==============================================================================
// intersect
//=============================================================================

template <class T>
HOSTDEV
void
testQuadForIntersections(um2::Quadrilateral2<T> const & quad)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto const aabb = quad.boundingBox();
  auto const longest_edge =
      aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<T>(rays_per_longest_edge);
  T const pi_deg = um2::pi_2<T> / static_cast<T>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    T const angle = pi_deg * static_cast<T>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      T buf[4];
      auto const hits = quad.intersect(ray, buf);
      // For each intersection coordinate
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2<T> const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        T min_dist = um2::infDistance<T>();
        for (Int ie = 0; ie < 4; ++ie) {
          um2::LineSegment<2, T> const l = quad.getEdge(ie);
          T const d = l.distanceTo(p);
          if (d < min_dist) {
            min_dist = d;
          }
        }
        ASSERT(min_dist < um2::epsDistance<T>());
      }
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(intersect)
{
  um2::Quadrilateral2<T> quad = makeQuad<2, T>();
  testQuadForIntersections(quad);
  quad = makeTriQuad<2, T>();
  testQuadForIntersections(quad);
}

template <Int D, class T>
TEST_SUITE(Quadrilateral)
{
  TEST_HOSTDEV(interpolate, D, T);
  TEST_HOSTDEV(jacobian, D, T);
  TEST_HOSTDEV(getEdge, D, T);
  TEST_HOSTDEV(perimeter, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isConvex, T);
    TEST_HOSTDEV(area, T);
    TEST_HOSTDEV(centroid, T);
    TEST_HOSTDEV(isCCW_flip, T);
    TEST_HOSTDEV(contains, T);
    TEST_HOSTDEV(meanChordLength, T);
    TEST_HOSTDEV(intersect, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((Quadrilateral<2, float>));
  RUN_SUITE((Quadrilateral<3, float>));

  RUN_SUITE((Quadrilateral<2, double>));
  RUN_SUITE((Quadrilateral<3, double>));
  return 0;
}
