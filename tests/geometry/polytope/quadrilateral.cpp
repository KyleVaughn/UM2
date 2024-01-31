#include <um2/geometry/polygon.hpp>

#include "../../test_macros.hpp"

F constexpr eps = um2::eps_distance * condCast<F>(10);

template <I D>
HOSTDEV constexpr auto
makeQuad() -> um2::Quadrilateral<D>
{
  um2::Quadrilateral<D> quad;
  for (I i = 0; i < 4; ++i) {
    for (I j = 0; j < D; ++j) {
      quad[i][j] = condCast<F>(0);
    }
  }
  quad[1][0] = condCast<F>(1);
  quad[2][0] = condCast<F>(1);
  quad[2][1] = condCast<F>(1);
  quad[3][1] = condCast<F>(1);
  return quad;
}

template <I D>
HOSTDEV constexpr auto
makeTriQuad() -> um2::Quadrilateral<D>
{
  um2::Quadrilateral<D> quad;
  for (I i = 0; i < 4; ++i) {
    for (I j = 0; j < D; ++j) {
      quad[i][j] = condCast<F>(0);
    }
  }
  quad[1][0] = condCast<F>(1);
  quad[2][0] = condCast<F>(1);
  quad[2][1] = condCast<F>(1);
  quad[3][1] = condCast<F>(0.5);
  quad[3][0] = condCast<F>(0.5);
  return quad;
}

//==============================================================================
// Interpolation
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Quadrilateral<D> quad = makeQuad<D>();
  um2::Point<D> const p00 = quad(0, 0);
  um2::Point<D> const p10 = quad(1, 0);
  um2::Point<D> const p01 = quad(0, 1);
  um2::Point<D> const p11 = quad(1, 1);
  ASSERT(um2::isApprox(p00, quad[0]));
  ASSERT(um2::isApprox(p10, quad[1]));
  ASSERT(um2::isApprox(p01, quad[3]));
  ASSERT(um2::isApprox(p11, quad[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::Quadrilateral<D> const quad = makeQuad<D>();
  auto jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), condCast<F>(1), eps);
  ASSERT_NEAR((jac(1, 0)), condCast<F>(0), eps);
  ASSERT_NEAR((jac(0, 1)), condCast<F>(0), eps);
  ASSERT_NEAR((jac(1, 1)), condCast<F>(1), eps);
  jac = quad.jacobian(condCast<F>(0.2), condCast<F>(0.3));
  ASSERT_NEAR((jac(0, 0)), condCast<F>(1), eps);
  ASSERT_NEAR((jac(1, 0)), condCast<F>(0), eps);
  ASSERT_NEAR((jac(0, 1)), condCast<F>(0), eps);
  ASSERT_NEAR((jac(1, 1)), condCast<F>(1), eps);
}

//==============================================================================
// edge
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(edge)
{
  um2::Quadrilateral<D> quad = makeQuad<D>();
  um2::LineSegment<D> edge = quad.getEdge(0);
  ASSERT(um2::isApprox(edge[0], quad[0]));
  ASSERT(um2::isApprox(edge[1], quad[1]));
  edge = quad.getEdge(1);
  ASSERT(um2::isApprox(edge[0], quad[1]));
  ASSERT(um2::isApprox(edge[1], quad[2]));
  edge = quad.getEdge(2);
  ASSERT(um2::isApprox(edge[0], quad[2]));
  ASSERT(um2::isApprox(edge[1], quad[3]));
  edge = quad.getEdge(3);
  ASSERT(um2::isApprox(edge[0], quad[3]));
  ASSERT(um2::isApprox(edge[1], quad[0]));
}

//==============================================================================
// isConvex
//==============================================================================

HOSTDEV
TEST_CASE(isConvex)
{
  um2::Quadrilateral<2> quad = makeQuad<2>();
  ASSERT(isConvex(quad));
  quad[3][0] = condCast<F>(0.5);
  ASSERT(isConvex(quad));
  quad[3][1] = condCast<F>(0.5);
  ASSERT(isConvex(quad)); // Effectively a triangle.
  quad[3][0] = condCast<F>(0.75);
  ASSERT(!isConvex(quad));
}

//==============================================================================
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  um2::Point2 p = um2::Point2(condCast<F>(0.25), condCast<F>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(condCast<F>(0.5), condCast<F>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2(condCast<F>(1.25), condCast<F>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2(condCast<F>(0.25), condCast<F>(-0.25));
  ASSERT(!quad.contains(p));
}

//==============================================================================
// area
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(area)
{
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  // Compiler has issues if we make this a static_assert.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(quad.area(), condCast<F>(1), eps);
  um2::Quadrilateral<2> const triquad = makeTriQuad<2>();
  ASSERT_NEAR(triquad.area(), condCast<F>(0.5), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// perimeter
//==============================================================================
template <I D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Quadrilateral<D> const quad = makeQuad<D>();
  ASSERT_NEAR(quad.perimeter(), condCast<F>(4), eps);
}

//==============================================================================
// centroid
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Quadrilateral<D> quad = makeQuad<D>();
  um2::Point<D> c = quad.centroid();
  ASSERT_NEAR(c[0], condCast<F>(0.5), eps);
  ASSERT_NEAR(c[1], condCast<F>(0.5), eps);
  quad[2] = um2::Point<D>(condCast<F>(2), condCast<F>(0.5));
  quad[3] = um2::Point<D>(condCast<F>(1), condCast<F>(0.5));
  c = quad.centroid();
  ASSERT_NEAR(c[0], condCast<F>(1.00), eps);
  ASSERT_NEAR(c[1], condCast<F>(0.25), eps);
  um2::Quadrilateral<D> const quad2 = makeTriQuad<D>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], condCast<F>(condCast<F>(2) / 3), eps);
  ASSERT_NEAR(c[1], condCast<F>(condCast<F>(1) / 3), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <I D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Quadrilateral<D> const quad = makeQuad<D>();
  um2::AxisAlignedBox<D> const box = quad.boundingBox();
  ASSERT_NEAR(box.minima()[0], condCast<F>(0), eps);
  ASSERT_NEAR(box.minima()[1], condCast<F>(0), eps);
  ASSERT_NEAR(box.maxima()[0], condCast<F>(1), eps);
  ASSERT_NEAR(box.maxima()[1], condCast<F>(1), eps);
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  um2::Quadrilateral<2> quad = makeQuad<2>();
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
  um2::Quadrilateral<2> const quad = makeQuad<2>();
  ASSERT_NEAR(quad.meanChordLength(), um2::pi_4<F>, eps);
}

#if UM2_USE_CUDA
template <I D>
MAKE_CUDA_KERNEL(interpolate, D);

template <I D>
MAKE_CUDA_KERNEL(jacobian, D);

template <I D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(isConvex);

MAKE_CUDA_KERNEL(contains);

template <I D>
MAKE_CUDA_KERNEL(area, D);

template <I D>
MAKE_CUDA_KERNEL(perimeter, D);

template <I D>
MAKE_CUDA_KERNEL(centroid, D);

template <I D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif

template <I D>
TEST_SUITE(Quadrilateral)
{
  TEST_HOSTDEV(interpolate, 1, 1, D);
  TEST_HOSTDEV(jacobian, 1, 1, D);
  TEST_HOSTDEV(edge, 1, 1, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isConvex);
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(isCCW_flipFace);
  }
  TEST_HOSTDEV(area, 1, 1, D);
  TEST_HOSTDEV(perimeter, 1, 1, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(centroid, 1, 1, D);
    TEST_HOSTDEV(meanChordLength);
  }
  TEST_HOSTDEV(boundingBox, 1, 1, D);
}

auto
main() -> int
{
  RUN_SUITE(Quadrilateral<2>);
  RUN_SUITE(Quadrilateral<3>);
  return 0;
}
