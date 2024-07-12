#include <um2/config.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/math/mat.hpp>
#include <um2/math/vec.hpp>

// NOLINTNEXTLINE(misc-include-cleaner)
#include <um2/geometry/quadratic_triangle.hpp>

#include "../../test_macros.hpp"

#include <cstdint>
#include <random>

template <class T>
T constexpr eps = um2::epsDistance<T>();

template <Int D, class T>
HOSTDEV constexpr auto
makeTri() -> um2::QuadraticTriangle<D, T>
{
  um2::QuadraticTriangle<D, T> this_tri;
  for (Int i = 0; i < 6; ++i) {
    this_tri[i] = 0;
  }
  this_tri[1][0] = castIfNot<T>(1);
  this_tri[2][1] = castIfNot<T>(1);
  this_tri[3][0] = castIfNot<T>(0.5);
  this_tri[4][0] = castIfNot<T>(0.5);
  this_tri[4][1] = castIfNot<T>(0.5);
  this_tri[5][1] = castIfNot<T>(0.5);
  return this_tri;
}

// P4 = (0.7, 0.8)
template <Int D, class T>
HOSTDEV constexpr auto
makeTri2() -> um2::QuadraticTriangle<D, T>
{
  um2::QuadraticTriangle<D, T> this_tri;
  for (Int i = 0; i < 6; ++i) {
    this_tri[i] = 0;
  }
  this_tri[1][0] = castIfNot<T>(1);
  this_tri[2][1] = castIfNot<T>(1);
  this_tri[3][0] = castIfNot<T>(0.5);
  this_tri[4][0] = castIfNot<T>(0.7);
  this_tri[4][1] = castIfNot<T>(0.8);
  this_tri[5][1] = castIfNot<T>(0.5);
  return this_tri;
}

template <class T>
HOSTDEV void
rotate(um2::QuadraticTriangle2<T> & q, T const angle)
{
  um2::Mat2x2<T> const rot = um2::makeRotationMatrix(angle);
  q[0] = rot * q[0];
  q[1] = rot * q[1];
  q[2] = rot * q[2];
  q[3] = rot * q[3];
  q[4] = rot * q[4];
  q[5] = rot * q[5];
}

template <class T>
void
perturb(um2::QuadraticTriangle2<T> & q)
{
  auto constexpr delta = castIfNot<T>(0.2);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  static std::mt19937 gen(seed);
  static std::uniform_real_distribution<T> dis(-delta, delta);
  for (Int i = 0; i < 6; ++i) {
    q[i][0] += dis(gen);
    q[i][1] += dis(gen);
  }
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticTriangle<D, T> tri = makeTri2<D, T>();
  ASSERT(tri(0, 0).isApprox(tri[0]));
  ASSERT(tri(1, 0).isApprox(tri[1]));
  ASSERT(tri(0, 1).isApprox(tri[2]));
  ASSERT(tri(0.5, 0).isApprox(tri[3]));
  ASSERT(tri(0.5, 0.5).isApprox(tri[4]));
  ASSERT(tri(0, 0.5).isApprox(tri[5]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::QuadraticTriangle<D, T> tri = makeTri<D, T>();
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
  um2::QuadraticTriangle<D, T> const tri = makeTri2<D, T>();
  um2::QuadraticSegment<D, T> edge = tri.getEdge(0);
  ASSERT(edge[0].isApprox(tri[0]));
  ASSERT(edge[1].isApprox(tri[1]));
  ASSERT(edge[2].isApprox(tri[3]));
  edge = tri.getEdge(1);
  ASSERT(edge[0].isApprox(tri[1]));
  ASSERT(edge[1].isApprox(tri[2]));
  ASSERT(edge[2].isApprox(tri[4]));
  edge = tri.getEdge(2);
  ASSERT(edge[0].isApprox(tri[2]));
  ASSERT(edge[1].isApprox(tri[0]));
  ASSERT(edge[2].isApprox(tri[5]));
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::QuadraticTriangle<D, T> const tri = makeTri<D, T>();
  // 1 + 1 + sqrt(2)
  ASSERT_NEAR(tri.perimeter(), castIfNot<T>(3.41421356), eps<T>);
}

//==============================================================================
// boundingBox
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticTriangle2<T> const tri = makeTri2<2, T>();
  auto const box = tri.boundingBox();
  // Actually making this a static assert causes a compiler error.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(box.minima(0), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.minima(1), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima(0), castIfNot<T>(1), eps<T>);
  ASSERT_NEAR(box.maxima(1), castIfNot<T>(1.0083333), eps<T>);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// area
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticTriangle<2, T> tri = makeTri<2, T>();
  um2::QuadraticTriangle<2, T> tri2 = makeTri2<2, T>();
  ASSERT_NEAR(tri.area(), castIfNot<T>(0.5), eps<T>);
  tri[3] = um2::Point2<T>(castIfNot<T>(0.5), castIfNot<T>(0.05));
  tri[5] = um2::Point2<T>(castIfNot<T>(0.05), castIfNot<T>(0.5));

  for (Int i = 0; i < 16; ++i) {
    rotate(tri, static_cast<T>(i) * um2::pi<T> / 8);
    rotate(tri2, static_cast<T>(i) * um2::pi<T> / 8);
    ASSERT_NEAR(tri.area(), castIfNot<T>(0.4333333333), eps<T>);
    ASSERT_NEAR(tri2.area(), castIfNot<T>(0.83333333), eps<T>);
  }
}

//==============================================================================
// centroid
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticTriangle<2, T> const tri = makeTri<2, T>();
  um2::Point<2, T> c = tri.centroid();
  ASSERT_NEAR(c[0], castIfNot<T>(1.0 / 3.0), eps<T>);
  ASSERT_NEAR(c[1], castIfNot<T>(1.0 / 3.0), eps<T>);

  um2::QuadraticTriangle<2, T> tri2 = makeTri2<2, T>();
  c = tri2.centroid();
  um2::Point2<T> ref(castIfNot<T>(0.432), castIfNot<T>(0.448));
  for (Int i = 0; i < 16; ++i) {
    T const angle = static_cast<T>(i) * um2::pi<T> / 8;
    rotate(tri2, angle);
    c = tri2.centroid();
    um2::Mat2x2<T> const rot = um2::makeRotationMatrix(angle);
    ref = rot * ref;
    ASSERT_NEAR(c[0], ref[0], eps<T>);
    ASSERT_NEAR(c[1], ref[1], eps<T>);
  }
}

//==============================================================================
// isCCW
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(isCCW_flip)
{
  auto tri = makeTri<2, T>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  um2::swap(tri[3], tri[5]);
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
  um2::QuadraticTriangle<2, T> const tri = makeTri2<2, T>();
  um2::Point2<T> p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.5), castIfNot<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(1.25), castIfNot<T>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.25), castIfNot<T>(-0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(castIfNot<T>(0.6), castIfNot<T>(0.6));
  ASSERT(tri.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

// This test used to be more meaningful, since the mean chord length was
// computed numerically.
template <class T>
HOSTDEV
TEST_CASE(meanChordLength)
{
  // Test convex
  auto const tri = makeTri<2, T>();
  auto const ref = um2::pi<T> * tri.area() / tri.perimeter();
  auto const val = tri.meanChordLength();
  auto const err = um2::abs(val - ref) / ref;
  // Relative error should be less than 0.1%.
  ASSERT(err < castIfNot<T>(1e-3));

  auto const tri2 = makeTri2<2, T>();
  auto const ref2 = um2::pi<T> * tri2.area() / tri2.perimeter();
  auto const val2 = tri2.meanChordLength();
  auto const err2 = um2::abs(val2 - ref2) / ref2;
  ASSERT(err2 < castIfNot<T>(1e-3));

  // A concave triangle
  um2::QuadraticTriangle<2, T> tri3 = makeTri<2, T>();
  tri3[4][0] = castIfNot<T>(0.25);
  tri3[4][1] = castIfNot<T>(0.25);
  auto const val3 = tri3.meanChordLength();
  auto const ref3 = um2::pi<T> * tri3.area() / tri3.perimeter();
  auto const err3 = um2::abs(val3 - ref3) / ref3;
  ASSERT(err3 < castIfNot<T>(1e-3));
}

//==============================================================================
// intersect
//=============================================================================

template <class T>
HOSTDEV void
testTriForIntersections(um2::QuadraticTriangle<2, T> const tri)
{
  // Parameters
  Int constexpr num_angles = 16; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 200;

  auto const aabb = tri.boundingBox();
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
    T buf[6];
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto const hits = tri.intersect(ray, buf);
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2<T> const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        T min_dist = um2::infDistance<T>();
        for (Int ie = 0; ie < 3; ++ie) {
          um2::QuadraticSegment<2, T> const q = tri.getEdge(ie);
          T const d = q.distanceTo(p);
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
  auto tri = makeTri<2, T>();
  testTriForIntersections(tri);
  tri = makeTri2<2, T>();
  testTriForIntersections(tri);
  tri[4][0] = castIfNot<T>(0.3);
  tri[4][1] = castIfNot<T>(0.25);
  testTriForIntersections(tri);

  for (Int ia = 0; ia < 16; ++ia) {
    for (Int ip = 0; ip < 5; ++ip) {
      tri = makeTri<2, T>();
      rotate(tri, static_cast<T>(ia) * um2::pi<T> / 8);
      perturb(tri);
      testTriForIntersections(tri);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(hasSelfIntersection)
{
  auto tri = makeTri<2, T>();
  ASSERT(!tri.hasSelfIntersection());
  tri[4][0] = castIfNot<T>(0.3);
  tri[4][1] = castIfNot<T>(0.25);
  ASSERT(!tri.hasSelfIntersection());

  {
    um2::Point2<T> const p0(castIfNot<T>(1.35267), castIfNot<T>(0.626669));
    um2::Point2<T> const p1(castIfNot<T>(1.39696), castIfNot<T>(0.798261));
    um2::Point2<T> const p2(castIfNot<T>(1.38913), castIfNot<T>(0.821739));
    um2::Point2<T> const p3(castIfNot<T>(1.37481), castIfNot<T>(0.712465));
    um2::Point2<T> const p4(castIfNot<T>(1.39304), castIfNot<T>(0.81));
    um2::Point2<T> const p5(castIfNot<T>(1.37994), castIfNot<T>(0.722514));
    tri[0] = p0;
    tri[1] = p1;
    tri[2] = p2;
    tri[3] = p3;
    tri[4] = p4;
    tri[5] = p5;
    ASSERT(tri.hasSelfIntersection());
  }

  {
    um2::Point2<T> const p0(castIfNot<T>(1.38913), castIfNot<T>(0));
    um2::Point2<T> const p1(castIfNot<T>(1.39696), castIfNot<T>(0));
    um2::Point2<T> const p2(castIfNot<T>(1.35656), castIfNot<T>(0.184691));
    um2::Point2<T> const p3(castIfNot<T>(1.39304), castIfNot<T>(0));
    um2::Point2<T> const p4(castIfNot<T>(1.37676), castIfNot<T>(0.0923454));
    um2::Point2<T> const p5(castIfNot<T>(1.38093), castIfNot<T>(0.09377));
    tri[0] = p0;
    tri[1] = p1;
    tri[2] = p2;
    tri[3] = p3;
    tri[4] = p4;
    tri[5] = p5;
    ASSERT(tri.hasSelfIntersection());
  }
}

template <class T>
HOSTDEV
TEST_CASE(fixSelfIntersection)
{
  auto tri = makeTri<2, T>();
  um2::Point2<T> const p0(castIfNot<T>(3.90250), castIfNot<T>(1.93887));
  um2::Point2<T> const p1(castIfNot<T>(3.90250), castIfNot<T>(1.95125));
  um2::Point2<T> const p2(castIfNot<T>(3.75839), castIfNot<T>(1.85233));
  um2::Point2<T> const p3(castIfNot<T>(3.90250), castIfNot<T>(1.94506));
  um2::Point2<T> const p4(castIfNot<T>(3.83044), castIfNot<T>(1.90179));
  um2::Point2<T> const p5(castIfNot<T>(3.82672), castIfNot<T>(1.90180));
  tri[0] = p0;
  tri[1] = p1;
  tri[2] = p2;
  tri[3] = p3;
  tri[4] = p4;
  tri[5] = p5;
  ASSERT(tri.hasSelfIntersection());
  Int vid = -1;
  um2::Point2<T> buffer[6];
  ASSERT(um2::fixSelfIntersection(tri, buffer, vid));
  ASSERT(vid == 4);
  ASSERT(!tri.hasSelfIntersection());
}

template <Int D, class T>
TEST_SUITE(QuadraticTriangle)
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
    TEST_HOSTDEV(hasSelfIntersection, T);
    TEST_HOSTDEV(fixSelfIntersection, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((QuadraticTriangle<2, float>));
  RUN_SUITE((QuadraticTriangle<3, float>));

  RUN_SUITE((QuadraticTriangle<2, double>));
  RUN_SUITE((QuadraticTriangle<3, double>));
  return 0;
}
