#include <um2/geometry/polygon.hpp>

#include "../../test_macros.hpp"

Float constexpr eps = um2::eps_distance * castIfNot<Float>(10);

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
  ASSERT(um2::isApprox(quad(0, 0), quad[0]));
  ASSERT(um2::isApprox(quad(1, 0), quad[1]));
  ASSERT(um2::isApprox(quad(1, 1), quad[2]));
  ASSERT(um2::isApprox(quad(0, 1), quad[3]));
  ASSERT(um2::isApprox(quad(0.5, 0), quad[4]));
  ASSERT(um2::isApprox(quad(1, 0.5), quad[5]));
  ASSERT(um2::isApprox(quad(0.5, 1), quad[6]));
  ASSERT(um2::isApprox(quad(0, 0.5), quad[7]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  // Floator the reference quad, the Jacobian is constant.
  um2::QuadraticQuadrilateral<D> quad = makeQuad<D>();
  auto jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
  jac = quad.jacobian(castIfNot<Float>(0.2), castIfNot<Float>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
  // If we stretch the quad, the Jacobian should change.
  quad[1][0] = castIfNot<Float>(2);
  jac = quad.jacobian(0.5, 0);
  ASSERT_NEAR((jac(0, 0)), 2, eps);
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
  ASSERT(um2::isApprox(edge[0], quad[0]));
  ASSERT(um2::isApprox(edge[1], quad[1]));
  ASSERT(um2::isApprox(edge[2], quad[4]));
  edge = quad.getEdge(1);
  ASSERT(um2::isApprox(edge[0], quad[1]));
  ASSERT(um2::isApprox(edge[1], quad[2]));
  ASSERT(um2::isApprox(edge[2], quad[5]));
  edge = quad.getEdge(2);
  ASSERT(um2::isApprox(edge[0], quad[2]));
  ASSERT(um2::isApprox(edge[1], quad[3]));
  ASSERT(um2::isApprox(edge[2], quad[6]));
  edge = quad.getEdge(3);
  ASSERT(um2::isApprox(edge[0], quad[3]));
  ASSERT(um2::isApprox(edge[1], quad[0]));
  ASSERT(um2::isApprox(edge[2], quad[7]));
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
TEST_CASE(isCCW_flipFace)
{
  auto quad = makeQuad<2>();
  ASSERT(quad.isCCW());
  um2::swap(quad[1], quad[3]);
  ASSERT(!quad.isCCW());
  um2::flipFace(quad);
  ASSERT(quad.isCCW());
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  auto const quad = makeQuad<2>();
  auto const ref = um2::pi<Float> / castIfNot<Float>(4);
  ASSERT_NEAR(quad.meanChordLength(), ref, castIfNot<Float>(3e-4));
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
#endif

template <Int D>
TEST_SUITE(QuadraticQuadrilateral)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(edge, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(area);
    TEST_HOSTDEV(centroid);
    TEST_HOSTDEV(boundingBox);
    TEST_HOSTDEV(isCCW_flipFace);
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
