#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/math/vec.hpp>

// NOLINTNEXTLINE(misc-include-cleaner)
#include <um2/geometry/quadratic_quadrilateral.hpp>

#include "../../test_macros.hpp"

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeQuad() -> um2::QuadraticQuadrilateral<D, T>
{
  um2::QuadraticQuadrilateral<D, T> this_quad;
  for (Int i = 0; i < 8; ++i) {
    this_quad[i] = 0;
  }
  this_quad[1][0] = castIfNot<T>(1);
  this_quad[2][0] = castIfNot<T>(1);
  this_quad[2][1] = castIfNot<T>(1);
  this_quad[3][1] = castIfNot<T>(1);
  this_quad[4][0] = castIfNot<T>(0.5);
  this_quad[5][0] = castIfNot<T>(1);
  this_quad[5][1] = castIfNot<T>(0.5);
  this_quad[6][0] = castIfNot<T>(0.5);
  this_quad[6][1] = castIfNot<T>(1);
  this_quad[7][1] = castIfNot<T>(0.5);
  return this_quad;
}

// P6 = (0.8, 1.5)
template <Int D, class T>
HOSTDEV constexpr auto
makeQuad2() -> um2::QuadraticQuadrilateral<D, T>
{
  um2::QuadraticQuadrilateral<D, T> this_quad = makeQuad<D, T>();
  this_quad[6][0] = castIfNot<T>(0.8);
  this_quad[6][1] = castIfNot<T>(1.5);
  return this_quad;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad2<D, T>();
  ASSERT(quad(0, 0).isApprox(quad[0]));
  ASSERT(quad(1, 0).isApprox(quad[1]));
  ASSERT(quad(1, 1).isApprox(quad[2]));
  ASSERT(quad(0, 1).isApprox(quad[3]));
  ASSERT(quad(0.5, 0).isApprox(quad[4]));
  ASSERT(quad(1, 0.5).isApprox(quad[5]));
  ASSERT(quad(0.5, 1).isApprox(quad[6]));
  ASSERT(quad(0, 0.5).isApprox(quad[7]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad<D, T>();
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
  // If we stretch the quad, the Jacobian should change.
  quad[1][0] = static_cast<T>(2);
  quad[2][0] = static_cast<T>(2);
  quad[5][0] = static_cast<T>(2);
  jac = quad.jacobian(0.5, 0);
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
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad2<D, T>();
  um2::QuadraticSegment<D, T> edge = quad.getEdge(0);
  ASSERT(edge[0].isApprox(quad[0]));
  ASSERT(edge[1].isApprox(quad[1]));
  ASSERT(edge[2].isApprox(quad[4]));
  edge = quad.getEdge(1);
  ASSERT(edge[0].isApprox(quad[1]));
  ASSERT(edge[1].isApprox(quad[2]));
  ASSERT(edge[2].isApprox(quad[5]));
  edge = quad.getEdge(2);
  ASSERT(edge[0].isApprox(quad[2]));
  ASSERT(edge[1].isApprox(quad[3]));
  ASSERT(edge[2].isApprox(quad[6]));
  edge = quad.getEdge(3);
  ASSERT(edge[0].isApprox(quad[3]));
  ASSERT(edge[1].isApprox(quad[0]));
  ASSERT(edge[2].isApprox(quad[7]));
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::QuadraticQuadrilateral<D, T> const quad = makeQuad<D, T>();
  ASSERT_NEAR(quad.perimeter(), castIfNot<T>(4), eps<T>);
}

//==============================================================================
// boundingBox
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad2<2, T>();
  um2::AxisAlignedBox<2, T> const box = quad.boundingBox();
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(box.minima(0), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.minima(1), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima(0), castIfNot<T>(1.0083333), eps<T>);
  ASSERT_NEAR(box.maxima(1), castIfNot<T>(1.5), eps<T>);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// area
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticQuadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT_NEAR(quad.area(), castIfNot<T>(1), eps<T>);
  quad[5] = um2::Point2<T>(castIfNot<T>(1.1), castIfNot<T>(0.5));
  quad[7] = um2::Point2<T>(castIfNot<T>(0.1), castIfNot<T>(0.5));
  ASSERT_NEAR(quad.area(), castIfNot<T>(1), eps<T>);

  um2::QuadraticQuadrilateral<2, T> const quad2 = makeQuad2<2, T>();
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(quad2.area(), castIfNot<T>(1.3333333), eps<T>);
}

//==============================================================================
// centroid
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad<2, T>();
  um2::Point<2, T> c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(0.5), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(0.5), eps<T>);

  um2::QuadraticQuadrilateral<2, T> const quad2 = makeQuad2<2, T>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(0.53), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(0.675), eps<T>);
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(isCCW_flip)
{
  auto quad = makeQuad<2, T>();
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
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad2<2, T>();
  um2::Point2<T> p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.5), castIfNot<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(2.25), castIfNot<T>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(-0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.8), castIfNot<T>(1.3));
  ASSERT(quad.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(meanChordLength)
{
  auto const quad = makeQuad<2, T>();
  auto const ref = um2::pi<T> * quad.area() / quad.perimeter();
  auto const val = quad.meanChordLength();
  auto const err = um2::abs(val - ref) / ref;
  // Relative error should be less than 0.1%.
  ASSERT(err < castIfNot<T>(1e-3));

  auto const quad2 = makeQuad2<2, T>();
  auto const ref2 = um2::pi<T> * quad2.area() / quad2.perimeter();
  auto const val2 = quad2.meanChordLength();
  auto const err2 = um2::abs(val2 - ref2) / ref2;
  ASSERT(err2 < castIfNot<T>(1e-3));

  // Non-convex quad
  auto quad3 = makeQuad<2, T>();
  quad3[4][0] = castIfNot<T>(0.7);
  quad3[4][1] = castIfNot<T>(0.25);
  auto const ref3 = um2::pi<T> * quad3.area() / quad3.perimeter();
  auto const val3 = quad3.meanChordLength();
  auto const err3 = um2::abs(val3 - ref3) / ref3;
  ASSERT(err3 < castIfNot<T>(1e-3));
}

//==============================================================================
// intersect
//=============================================================================

template <class T>
HOSTDEV void
testQuadForIntersections(um2::QuadraticQuadrilateral<2, T> const quad)
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
      T buf[8];
      auto const hits = quad.intersect(ray, buf);
      // For each intersection coordinate
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2<T> const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        T min_dist = um2::infDistance<T>();
        for (Int ie = 0; ie < 4; ++ie) {
          um2::QuadraticSegment<2, T> const q = quad.getEdge(ie);
          T const d = q.distanceTo(p);
          if (d < min_dist) {
            min_dist = d;
          }
        }
        // Check if the distance is close to zero
        ASSERT(min_dist < eps<T>);
      }
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(intersect)
{
  auto quad = makeQuad<2, T>();
  testQuadForIntersections(quad);
  quad = makeQuad2<2, T>();
  testQuadForIntersections(quad);
}

template <Int D, class T>
TEST_SUITE(QuadraticQuadrilateral)
{
  TEST_HOSTDEV(interpolate, D, T);
  TEST_HOSTDEV(jacobian, D, T);
  TEST_HOSTDEV(getEdge, D, T);
  TEST_HOSTDEV(perimeter, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(boundingBox, T);
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
  RUN_SUITE((QuadraticQuadrilateral<2, float>));
  RUN_SUITE((QuadraticQuadrilateral<3, float>));

  RUN_SUITE((QuadraticQuadrilateral<2, double>));
  RUN_SUITE((QuadraticQuadrilateral<3, double>));
  return 0;
}
