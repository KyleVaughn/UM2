#include <um2/geometry/quadratic_quadrilateral.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>

#include "../../test_macros.hpp"

Float constexpr eps = um2::eps_distance;

template <Int D>
HOSTDEV constexpr auto
makeQuad() -> um2::QuadraticQuadrilateral<D>
{
  um2::QuadraticQuadrilateral<D> this_quad;
  for (Int i = 0; i < 8; ++i) {
    this_quad[i] = um2::Vec<D, Float>::zero();
  }
  this_quad[1][0] = castIfNot<Float>(1);
  this_quad[2][0] = castIfNot<Float>(1);
  this_quad[2][1] = castIfNot<Float>(1);
  this_quad[3][1] = castIfNot<Float>(1);
  this_quad[4][0] = castIfNot<Float>(0.5);
  this_quad[5][0] = castIfNot<Float>(1);
  this_quad[5][1] = castIfNot<Float>(0.5);
  this_quad[6][0] = castIfNot<Float>(0.5);
  this_quad[6][1] = castIfNot<Float>(1);
  this_quad[7][1] = castIfNot<Float>(0.5);
  return this_quad;
}

// P6 = (0.8, 1.5)
template <Int D>
HOSTDEV constexpr auto
makeQuad2() -> um2::QuadraticQuadrilateral<D>
{
  um2::QuadraticQuadrilateral<D> this_quad = makeQuad<D>();
  this_quad[6][0] = castIfNot<Float>(0.8);
  this_quad[6][1] = castIfNot<Float>(1.5);
  return this_quad;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticQuadrilateral<D> quad = makeQuad2<D>();
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
// edge
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(edge)
{
  um2::QuadraticQuadrilateral<D> quad = makeQuad2<D>();
  um2::QuadraticSegment<D> edge = quad.getEdge(0);
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
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::QuadraticQuadrilateral<2> const quad = makeQuad2<2>();
  um2::Point2 p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(castIfNot<Float>(2.25), castIfNot<Float>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(-0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2(castIfNot<Float>(0.8), castIfNot<Float>(1.3));
  ASSERT(quad.contains(p));
}

//==============================================================================
// area
//==============================================================================

HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticQuadrilateral<2> quad = makeQuad<2>();
  ASSERT_NEAR(quad.area(), castIfNot<Float>(1), eps);
  quad[5] = um2::Point2(castIfNot<Float>(1.1), castIfNot<Float>(0.5));
  quad[7] = um2::Point2(castIfNot<Float>(0.1), castIfNot<Float>(0.5));
  ASSERT_NEAR(quad.area(), castIfNot<Float>(1), eps);

  um2::QuadraticQuadrilateral<2> const quad2 = makeQuad2<2>();
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(quad2.area(), castIfNot<Float>(1.3333333), eps);
}

//==============================================================================
// perimeter
//==============================================================================

HOSTDEV
TEST_CASE(perimeter)
{
  um2::QuadraticQuadrilateral<2> const quad = makeQuad<2>();
  ASSERT_NEAR(quad.perimeter(), castIfNot<Float>(4), eps);
}

//==============================================================================
// centroid
//==============================================================================

HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticQuadrilateral<2> const quad = makeQuad<2>();
  um2::Point<2> c = quad.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(0.5), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(0.5), eps);

  um2::QuadraticQuadrilateral<2> const quad2 = makeQuad2<2>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(0.53), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(0.675), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticQuadrilateral<2> const quad = makeQuad2<2>();
  um2::AxisAlignedBox<2> const box = quad.boundingBox();
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(box.minima()[0], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima()[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], castIfNot<Float>(1.0083333), eps);
  ASSERT_NEAR(box.maxima()[1], castIfNot<Float>(1.5), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flip)
{
  auto quad = makeQuad<2>();
  ASSERT(quad.isCCW());
  um2::swap(quad[1], quad[3]);
  ASSERT(!quad.isCCW());
  quad.flip();
  ASSERT(quad.isCCW());
}

//==============================================================================
// intersect
//=============================================================================

HOSTDEV
void
testQuadForIntersections(um2::QuadraticQuadrilateral<2> const quad)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto const aabb = quad.boundingBox();
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
      auto const intersections = quad.intersect(ray);
      ASSERT(um2::is_sorted(intersections.begin(), intersections.end()));
      // For each intersection coordinate
      for (auto const & r : intersections) {
        // If intersection is valid
        if (0 <= r) {
          um2::Point2 const p = ray(r);
          // Get the distance to the closest edge
          Float min_dist = um2::inf_distance;
          for (Int ie = 0; ie < 4; ++ie) {
            um2::QuadraticSegment<2> const q = quad.getEdge(ie);
            Float const d = q.distanceTo(p);
            if (d < min_dist) {
              min_dist = d;
            }
          }
          // Check if the distance is close to zero
          ASSERT(min_dist < um2::eps_distance);
        }
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  auto quad = makeQuad<2>();
  testQuadForIntersections(quad);
  quad = makeQuad2<2>();
  testQuadForIntersections(quad);
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  auto const quad = makeQuad<2>();
  auto const ref = um2::pi<Float> * quad.area() / quad.perimeter();
  auto const val = quad.meanChordLength();
  auto const err = um2::abs(val - ref) / ref;
  // Relative error should be less than 0.1%.
  ASSERT(err < castIfNot<Float>(1e-3));

  auto const quad2 = makeQuad2<2>();
  auto const ref2 = um2::pi<Float> * quad2.area() / quad2.perimeter();
  auto const val2 = quad2.meanChordLength();
  auto const err2 = um2::abs(val2 - ref2) / ref2;
  ASSERT(err2 < castIfNot<Float>(1e-3));

  // Non-convex quad
  auto quad3 = makeQuad<2>();
  quad3[4][0] = castIfNot<Float>(0.7);
  quad3[4][1] = castIfNot<Float>(0.25);
  ASSERT(!quad3.isConvex());
  auto const ref3 = um2::pi<Float> * quad3.area() / quad3.perimeter();
  auto const val3 = quad3.meanChordLength();
  auto const err3 = um2::abs(val3 - ref3) / ref3;
  ASSERT(err3 < castIfNot<Float>(1e-3));
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Int D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(contains);

MAKE_CUDA_KERNEL(area);

MAKE_CUDA_KERNEL(centroid);

MAKE_CUDA_KERNEL(boundingBox);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif

template <Int D>
TEST_SUITE(QuadraticQuadrilateral)
{
  TEST_HOSTDEV(interpolate, D);
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
  RUN_SUITE(QuadraticQuadrilateral<2>);
  RUN_SUITE(QuadraticQuadrilateral<3>);
  return 0;
}
