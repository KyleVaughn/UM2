#include <um2/config.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/geometry/point.hpp>

// NOLINTNEXTLINE(misc-include-cleaner)
#include <um2/geometry/triangle.hpp>

#include "../../test_macros.hpp"

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeTri() -> um2::Triangle<D, T>
{
  um2::Triangle<D, T> this_tri;
  this_tri[0] = 0;
  this_tri[1] = 0;
  this_tri[2] = 0;
  this_tri[1][0] = castIfNot<T>(1);
  this_tri[2][1] = castIfNot<T>(1);
  return this_tri;
}

//==============================================================================
// interpolation
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Point<D, T> const p00 = tri(0, 0);
  um2::Point<D, T> const p10 = tri(1, 0);
  um2::Point<D, T> const p01 = tri(0, 1);
  ASSERT(p00.isApprox(tri[0]));
  ASSERT(p10.isApprox(tri[1]));
  ASSERT(p01.isApprox(tri[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::Triangle<D, T> tri = makeTri<D, T>();
  auto jac = tri.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps<T>);
  ASSERT_NEAR((jac(1, 0)), 0, eps<T>);
  ASSERT_NEAR((jac(0, 1)), 0, eps<T>);
  ASSERT_NEAR((jac(1, 1)), 1, eps<T>);
  jac = tri.jacobian(castIfNot<T>(0.2), castIfNot<T>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps<T>);
  ASSERT_NEAR((jac(1, 0)), 0, eps<T>);
  ASSERT_NEAR((jac(0, 1)), 0, eps<T>);
  ASSERT_NEAR((jac(1, 1)), 1, eps<T>);
  // If we stretch the triangle, the Jacobian should change.
  tri[1][0] = castIfNot<T>(2);
  jac = tri.jacobian(0.5, 0);
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
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::LineSegment<D, T> edge = tri.getEdge(0);
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

template <Int D, class T>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Triangle<D, T> const tri = makeTri<D, T>();
  auto const two = castIfNot<T>(2);
  T const ref = two + um2::sqrt(two);
  ASSERT_NEAR(tri.perimeter(), ref, eps<T>);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Triangle<D, T> const tri = makeTri<D, T>();
  um2::AxisAlignedBox<D, T> const box = tri.boundingBox();
  ASSERT_NEAR(box.minima()[0], castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.minima()[1], castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima()[0], castIfNot<T>(1), eps<T>);
  ASSERT_NEAR(box.maxima()[1], castIfNot<T>(1), eps<T>);
}

//==============================================================================
// area
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(area)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  ASSERT_NEAR(tri.area(), castIfNot<T>(0.5), eps<T>);
  tri[1][0] = castIfNot<T>(2);
  ASSERT_NEAR(tri.area(), castIfNot<T>(1), eps<T>);
}

//==============================================================================
// centroid
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Triangle<D, T> const tri = makeTri<D, T>();
  um2::Point<D, T> const c = tri.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(1.0 / 3.0), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(1.0 / 3.0), eps<T>);
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(isCCW_flip)
{
  um2::Triangle<2, T> tri = makeTri<2, T>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  ASSERT(!tri.isCCW());
  tri.flip();
  ASSERT(tri.isCCW());
}

//==============================================================================
// contains
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(contains)
{
  um2::Triangle<2, T> const tri = makeTri<2, T>();
  um2::Point2<T> p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.5), castIfNot<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(1.25), castIfNot<T>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(-0.25));
  ASSERT(!tri.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(meanChordLength)
{
  um2::Triangle<2, T> const tri = makeTri<2, T>();
  auto const two = castIfNot<T>(2);
  auto const ref = um2::pi<T> / (two * (two + um2::sqrt(two)));
  ASSERT_NEAR(tri.meanChordLength(), ref, eps<T>);
}

//==============================================================================
// intersect
//=============================================================================

template <class T>
HOSTDEV
void
testTriForIntersections(um2::Triangle<2, T> const tri)
{
  // Parameters
  Int constexpr num_angles = 64; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto aabb = tri.boundingBox();
  aabb.scale(castIfNot<T>(1.1));
  auto const longest_edge =
      aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<T>(rays_per_longest_edge);
  T const pi_deg = um2::pi_2<T> / static_cast<T>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    T const angle = pi_deg * static_cast<T>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams<T> const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      T buf[3];
      auto const hits = tri.intersect(ray, buf);
      // For each intersection coordinate
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2<T> const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        T min_dist = um2::infDistance<T>();
        for (Int ie = 0; ie < 3; ++ie) {
          um2::LineSegment<2, T> const l = tri.getEdge(ie);
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
  um2::Triangle2<T> tri = makeTri<2, T>();
  testTriForIntersections(tri);
  tri[1][0] = castIfNot<T>(2);
  testTriForIntersections(tri);
}

template <Int D, class T>
TEST_SUITE(Triangle)
{
  TEST_HOSTDEV(interpolate, D, T);
  TEST_HOSTDEV(jacobian, D, T);
  TEST_HOSTDEV(getEdge, D, T);
  TEST_HOSTDEV(perimeter, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  TEST_HOSTDEV(area, D, T);
  TEST_HOSTDEV(centroid, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isCCW_flip, T);
    TEST_HOSTDEV(contains, T);
    TEST_HOSTDEV(meanChordLength, T);
    TEST_HOSTDEV(intersect, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((Triangle<2, float>));
  RUN_SUITE((Triangle<3, float>));

  RUN_SUITE((Triangle<2, double>));
  RUN_SUITE((Triangle<3, double>));
  return 0;
}
