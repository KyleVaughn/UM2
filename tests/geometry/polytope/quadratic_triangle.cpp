#include <um2/geometry/quadratic_triangle.hpp>

#include "../../test_macros.hpp"

#include <iostream>

Float constexpr eps = um2::eps_distance * castIfNot<Float>(10);

template <Int D>
HOSTDEV constexpr auto
makeTri() -> um2::QuadraticTriangle<D>
{
  um2::QuadraticTriangle<D> this_tri;
  for (Int i = 0; i < 6; ++i) {
    this_tri[i] = um2::Vec<D, Float>::zero();
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
    this_tri[i] = um2::Vec<D, Float>::zero();
  }
  this_tri[1][0] = castIfNot<Float>(1);
  this_tri[2][1] = castIfNot<Float>(1);
  this_tri[3][0] = castIfNot<Float>(0.5);
  this_tri[4][0] = castIfNot<Float>(0.7);
  this_tri[4][1] = castIfNot<Float>(0.8);
  this_tri[5][1] = castIfNot<Float>(0.5);
  return this_tri;
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
  // Floator the reference triangle, the Jacobian is constant.
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
// edge
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(edge)
{
  um2::QuadraticTriangle<D> tri = makeTri2<D>();
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
// area
//==============================================================================

HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticTriangle<2> tri = makeTri<2>();
  ASSERT_NEAR(tri.area(), castIfNot<Float>(0.5), eps);
  tri[3] = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.05));
  tri[5] = um2::Point2(castIfNot<Float>(0.05), castIfNot<Float>(0.5));
  // Actually making this a static assert causes a compiler error.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(tri.area(), castIfNot<Float>(0.4333333333), eps);

  um2::QuadraticTriangle<2> const tri2 = makeTri2<2>();
  ASSERT_NEAR(tri2.area(), castIfNot<Float>(0.83333333), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// perimeter
//==============================================================================

HOSTDEV
TEST_CASE(perimeter)
{
  um2::QuadraticTriangle<2> const tri = makeTri<2>();
  // 1 + 1 + sqrt(2)
  ASSERT_NEAR(tri.perimeter(), castIfNot<Float>(3.41421356), eps);
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

  um2::QuadraticTriangle<2> const tri2 = makeTri2<2>();
  c = tri2.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(0.432), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(0.448), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticTriangle<2> const tri = makeTri2<2>();
  um2::AxisAlignedBox<2> const box = tri.boundingBox();
  // Actually making this a static assert causes a compiler error.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(box.minima()[0], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima()[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima()[1], castIfNot<Float>(1.008333), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
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
// intersect
//=============================================================================

HOSTDEV
void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
testTriForIntersections(um2::QuadraticTriangle<2> const tri)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 100;
  
  auto const aabb = tri.boundingBox();
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
      auto const intersections = tri.intersect(ray);
      // For each intersection coordinate
      for (auto const & r : intersections) {
        // If intersection is valid
        if (r < um2::inf_distance / 10) {
          um2::Point2 const p = ray(r);
          // Get the distance to the closest edge
          Float min_dist = um2::inf_distance;
          for (Int ie = 0; ie < 3; ++ie) {
            um2::QuadraticSegment<2> const q = tri.getEdge(ie);
            Float const d = q.distanceTo(p);
            if (d < min_dist) {
              min_dist = d;
            }
          }
          // Check if the distance is close to zero
          if (min_dist > 10 * um2::eps_distance) {
            std::cerr << "d = " << min_dist << std::endl;
            std::cerr << "r = " << r << std::endl;
            std::cerr << "p = (" << p[0] << ", " << p[1] << ")" << std::endl;
          }
          ASSERT(min_dist < 10 * um2::eps_distance);
        }
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
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  // Test convex
  auto const tri = makeTri<2>();
  auto const ref = um2::pi<Float> * tri.area() / tri.perimeter();
  auto const val = tri.meanChordLength();
  auto const err = um2::abs(val - ref) / ref;
  std::cerr << "val = " << val << std::endl;
  std::cerr << "ref = " << ref << std::endl;
  std::cerr << "err = " << err << std::endl;
  // Relative error should be less than 0.1%.
  ASSERT(err < castIfNot<Float>(1e-3));

  auto const tri2 = makeTri2<2>();
  auto const ref2 = um2::pi<Float> * tri2.area() / tri2.perimeter();
  auto const val2 = tri2.meanChordLength();
  auto const err2 = um2::abs(val2 - ref2) / ref2;
  std::cerr << "val2 = " << val2 << std::endl;
  std::cerr << "ref2 = " << ref2 << std::endl;
  std::cerr << "err2 = " << err2 << std::endl;
  ASSERT(err2 < castIfNot<Float>(1e-3));

  // A concave triangle
  um2::QuadraticTriangle<2> tri3 = makeTri<2>();
  tri3[4][0] = castIfNot<Float>(0.25);
  tri3[4][1] = castIfNot<Float>(0.25);
  ASSERT(!tri3.isConvex());
  auto const val3 = tri3.meanChordLength(); 
  auto const ref3 = um2::pi<Float> * tri3.area() / tri3.perimeter();
  auto const err3 = um2::abs(val3 - ref3) / ref3;
  std::cerr << "val3 = " << val3 << std::endl;
  std::cerr << "ref3 = " << ref3 << std::endl;
  std::cerr << "err3 = " << err3 << std::endl;
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Int D>
MAKE_CUDA_KERNEL(jacobian, D);

template <Int D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(contains);

MAKE_CUDA_KERNEL(area);

MAKE_CUDA_KERNEL(centroid);

MAKE_CUDA_KERNEL(boundingBox);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif // UM2_USE_CUDA

template <Int D>
TEST_SUITE(QuadraticTriangle)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(edge, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(area);
    TEST_HOSTDEV(perimeter);
    TEST_HOSTDEV(centroid);
    TEST_HOSTDEV(boundingBox);
    TEST_HOSTDEV(isCCW_flip);
    TEST_HOSTDEV(intersect);
    TEST_HOSTDEV(meanChordLength);
  }
}

auto
main() -> int
{
  RUN_SUITE(QuadraticTriangle<2>);
  RUN_SUITE(QuadraticTriangle<3>);
  return 0;
}
