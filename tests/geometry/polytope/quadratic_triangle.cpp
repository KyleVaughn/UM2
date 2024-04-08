#include <um2/geometry/quadratic_triangle.hpp>
#include <um2/geometry/modular_rays.hpp>

#include "../../test_macros.hpp"

#include <random>

Float constexpr eps = um2::eps_distance;

template <Int D>
HOSTDEV constexpr auto
makeTri() -> um2::QuadraticTriangle<D>
{
  um2::QuadraticTriangle<D> this_tri;
  for (Int i = 0; i < 6; ++i) {
    this_tri[i] = 0;
  }
  this_tri[1][0] = castIfNot<Float>(1);
  this_tri[2][1] = castIfNot<Float>(1);
  this_tri[3][0] = castIfNot<Float>(0.5);
  this_tri[4][0] = castIfNot<Float>(0.5);
  this_tri[4][1] = castIfNot<Float>(0.5);
  this_tri[5][1] = castIfNot<Float>(0.5);
  return this_tri;
}

// P4 = (0.7, 0.8)
template <Int D>
HOSTDEV constexpr auto
makeTri2() -> um2::QuadraticTriangle<D>
{
  um2::QuadraticTriangle<D> this_tri;
  for (Int i = 0; i < 6; ++i) {
    this_tri[i] = 0; 
  }
  this_tri[1][0] = castIfNot<Float>(1);
  this_tri[2][1] = castIfNot<Float>(1);
  this_tri[3][0] = castIfNot<Float>(0.5);
  this_tri[4][0] = castIfNot<Float>(0.7);
  this_tri[4][1] = castIfNot<Float>(0.8);
  this_tri[5][1] = castIfNot<Float>(0.5);
  return this_tri;
}

HOSTDEV void
rotate(um2::QuadraticTriangle2 & q, Float const angle)
{
  um2::Mat2x2F const rot = um2::makeRotationMatrix(angle);
  q[0] = rot * q[0];
  q[1] = rot * q[1];
  q[2] = rot * q[2];
  q[3] = rot * q[3];
  q[4] = rot * q[4];
  q[5] = rot * q[5];
}

void
perturb(um2::QuadraticTriangle2 & q)
{
  auto constexpr delta = castIfNot<Float>(0.2);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  static std::mt19937 gen(seed);
  static std::uniform_real_distribution<Float> dis(-delta, delta);
  for (Int i = 0; i < 6; ++i) {
    q[i][0] += dis(gen);
    q[i][1] += dis(gen);
  }
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticTriangle<D> tri = makeTri2<D>();
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

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::QuadraticTriangle<D> tri = makeTri<D>();
  auto jac = tri.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);

  jac = tri.jacobian(castIfNot<Float>(0.2), castIfNot<Float>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);

  // If we stretch the triangle, the Jacobian should change.
  tri[1][0] = castIfNot<Float>(2);
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
  um2::QuadraticTriangle<D> const tri = makeTri2<D>();
  um2::QuadraticSegment<D> edge = tri.getEdge(0);
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

template <Int D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::QuadraticTriangle<D> const tri = makeTri<D>();
  // 1 + 1 + sqrt(2)
  ASSERT_NEAR(tri.perimeter(), castIfNot<Float>(3.41421356), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticTriangle2 const tri = makeTri2<2>();
  auto const box = tri.boundingBox();
  // Actually making this a static assert causes a compiler error.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(box.minima(0), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima(1), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima(0), castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima(1), castIfNot<Float>(1.008333), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// area
//==============================================================================

HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticTriangle<2> tri = makeTri<2>();
  um2::QuadraticTriangle<2> tri2 = makeTri2<2>();
  ASSERT_NEAR(tri.area(), castIfNot<Float>(0.5), eps);
  tri[3] = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.05));
  tri[5] = um2::Point2(castIfNot<Float>(0.05), castIfNot<Float>(0.5));

  for (Int i = 0; i < 16; ++i) {
    rotate(tri, static_cast<Float>(i) * um2::pi<Float> / 8);
    rotate(tri2, static_cast<Float>(i) * um2::pi<Float> / 8);
    ASSERT_NEAR(tri.area(), castIfNot<Float>(0.4333333333), eps);
    ASSERT_NEAR(tri2.area(), castIfNot<Float>(0.83333333), eps);
  }
}

//==============================================================================
// centroid
//==============================================================================

HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticTriangle<2> const tri = makeTri<2>();
  um2::Point<2> c = tri.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(1.0 / 3.0), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(1.0 / 3.0), eps);

  um2::QuadraticTriangle<2> tri2 = makeTri2<2>();
  c = tri2.centroid();
  um2::Point2 ref(castIfNot<Float>(0.432), castIfNot<Float>(0.448));
  for (Int i = 0; i < 16; ++i) {
    Float const angle = static_cast<Float>(i) * um2::pi<Float> / 8;
    rotate(tri2, angle);
    c = tri2.centroid();
    um2::Mat2x2F const rot = um2::makeRotationMatrix(angle);
    ref = rot * ref;
    ASSERT_NEAR(c[0], ref[0], eps);
    ASSERT_NEAR(c[1], ref[1], eps);
  }
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flip)
{
  auto tri = makeTri<2>();
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

HOSTDEV
TEST_CASE(contains)
{
  um2::QuadraticTriangle<2> const tri = makeTri2<2>();
  um2::Point2 p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(1.25), castIfNot<Float>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(-0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.6), castIfNot<Float>(0.6));
  ASSERT(tri.contains(p));
}

//==============================================================================
// meanChordLength
//==============================================================================

// This test used to be more meaningful, since the mean chord length was
// computed numerically.
HOSTDEV
TEST_CASE(meanChordLength)
{
  // Test convex
  auto const tri = makeTri<2>();
  auto const ref = um2::pi<Float> * tri.area() / tri.perimeter();
  auto const val = tri.meanChordLength();
  auto const err = um2::abs(val - ref) / ref;
  // Relative error should be less than 0.1%.
  ASSERT(err < castIfNot<Float>(1e-3));

  auto const tri2 = makeTri2<2>();
  auto const ref2 = um2::pi<Float> * tri2.area() / tri2.perimeter();
  auto const val2 = tri2.meanChordLength();
  auto const err2 = um2::abs(val2 - ref2) / ref2;
  ASSERT(err2 < castIfNot<Float>(1e-3));

  // A concave triangle
  um2::QuadraticTriangle<2> tri3 = makeTri<2>();
  tri3[4][0] = castIfNot<Float>(0.25);
  tri3[4][1] = castIfNot<Float>(0.25);
  auto const val3 = tri3.meanChordLength();
  auto const ref3 = um2::pi<Float> * tri3.area() / tri3.perimeter();
  auto const err3 = um2::abs(val3 - ref3) / ref3;
  ASSERT(err3 < castIfNot<Float>(1e-3));
}

//==============================================================================
// intersect
//=============================================================================

HOSTDEV
void
testTriForIntersections(um2::QuadraticTriangle<2> const tri)
{
  // Parameters
  Int constexpr num_angles = 16; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 200;

  auto const aabb = tri.boundingBox();
  auto const longest_edge = aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    Float buf[6];
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto const hits = tri.intersect(ray, buf);
      for (Int ihit = 0; ihit < hits; ++ihit) {
        um2::Point2 const p = ray(buf[ihit]);
        // Get the distance to the closest edge
        Float min_dist = um2::inf_distance;
        for (Int ie = 0; ie < 3; ++ie) {
          um2::QuadraticSegment<2> const q = tri.getEdge(ie);
          Float const d = q.distanceTo(p);
          if (d < min_dist) {
            min_dist = d;
          }
        }
#if UM2_ENABLE_FLOAT64
        ASSERT(min_dist < um2::eps_distance);
#else
        ASSERT(min_dist < 20 * um2::eps_distance);
#endif
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  auto tri = makeTri<2>();
  testTriForIntersections(tri);
  tri = makeTri2<2>();
  testTriForIntersections(tri);
  tri[4][0] = castIfNot<Float>(0.3);
  tri[4][1] = castIfNot<Float>(0.25);
  testTriForIntersections(tri);

  for (Int ia = 0; ia < 16; ++ia) {
    for (Int ip = 0; ip < 5; ++ip) {
      tri = makeTri<2>();
      rotate(tri, static_cast<Float>(ia) * um2::pi<Float> / 8);
      perturb(tri);
      testTriForIntersections(tri);
    }
  }
}

template <Int D>
TEST_SUITE(QuadraticTriangle)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(getEdge, D);
  TEST_HOSTDEV(perimeter, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(boundingBox);
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
  RUN_SUITE(QuadraticTriangle<2>);
  RUN_SUITE(QuadraticTriangle<3>);
  return 0;
}
