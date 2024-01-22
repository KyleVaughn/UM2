#include <um2/geometry/polygon.hpp>

#include "../../test_macros.hpp"

// Ignore useless casts on initialization of points
// Point(static_cast<D>(0.1), static_cast<F>(0.2)) is not worth addressing
#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

F constexpr eps = um2::eps_distance * static_cast<F>(10);

template <Size D>
HOSTDEV constexpr auto
makeTri() -> um2::QuadraticTriangle<D>
{
  um2::QuadraticTriangle<D> this_tri;
  for (Size i = 0; i < 6; ++i) {
    this_tri[i] = um2::Vec<D, F>::zero();
  }
  this_tri[1][0] = static_cast<F>(1);
  this_tri[2][1] = static_cast<F>(1);
  this_tri[3][0] = static_cast<F>(0.5);
  this_tri[4][0] = static_cast<F>(0.5);
  this_tri[4][1] = static_cast<F>(0.5);
  this_tri[5][1] = static_cast<F>(0.5);
  return this_tri;
}

// P4 = (0.7, 0.8)
template <Size D>
HOSTDEV constexpr auto
makeTri2() -> um2::QuadraticTriangle<D>
{
  um2::QuadraticTriangle<D> this_tri;
  for (Size i = 0; i < 6; ++i) {
    this_tri[i] = um2::Vec<D, F>::zero();
  }
  this_tri[1][0] = static_cast<F>(1);
  this_tri[2][1] = static_cast<F>(1);
  this_tri[3][0] = static_cast<F>(0.5);
  this_tri[4][0] = static_cast<F>(0.7);
  this_tri[4][1] = static_cast<F>(0.8);
  this_tri[5][1] = static_cast<F>(0.5);
  return this_tri;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticTriangle<D> tri = makeTri2<D>();
  ASSERT(um2::isApprox(tri(0, 0), tri[0]));
  ASSERT(um2::isApprox(tri(1, 0), tri[1]));
  ASSERT(um2::isApprox(tri(0, 1), tri[2]));
  ASSERT(um2::isApprox(tri(0.5, 0), tri[3]));
  ASSERT(um2::isApprox(tri(0.5, 0.5), tri[4]));
  ASSERT(um2::isApprox(tri(0, 0.5), tri[5]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D>
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
  jac = tri.jacobian(static_cast<F>(0.2), static_cast<F>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
  // If we stretch the triangle, the Jacobian should change.
  tri[1][0] = static_cast<F>(2);
  jac = tri.jacobian(0.5, 0);
  ASSERT_NEAR((jac(0, 0)), 2, eps);
  ASSERT_NEAR((jac(1, 0)), 0, eps);
  ASSERT_NEAR((jac(0, 1)), 0, eps);
  ASSERT_NEAR((jac(1, 1)), 1, eps);
}

//==============================================================================
// edge
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(edge)
{
  um2::QuadraticTriangle<D> tri = makeTri2<D>();
  um2::QuadraticSegment<D> edge = tri.getEdge(0);
  ASSERT(um2::isApprox(edge[0], tri[0]));
  ASSERT(um2::isApprox(edge[1], tri[1]));
  ASSERT(um2::isApprox(edge[2], tri[3]));
  edge = tri.getEdge(1);
  ASSERT(um2::isApprox(edge[0], tri[1]));
  ASSERT(um2::isApprox(edge[1], tri[2]));
  ASSERT(um2::isApprox(edge[2], tri[4]));
  edge = tri.getEdge(2);
  ASSERT(um2::isApprox(edge[0], tri[2]));
  ASSERT(um2::isApprox(edge[1], tri[0]));
  ASSERT(um2::isApprox(edge[2], tri[5]));
}

//==============================================================================
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::QuadraticTriangle<2> const tri = makeTri2<2>();
  um2::Point2 p = um2::Point2(static_cast<F>(0.25), static_cast<F>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(static_cast<F>(0.5), static_cast<F>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(static_cast<F>(1.25), static_cast<F>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(static_cast<F>(0.25), static_cast<F>(-0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(static_cast<F>(0.6), static_cast<F>(0.6));
  ASSERT(tri.contains(p));
}

//==============================================================================
// area
//==============================================================================

HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticTriangle<2> tri = makeTri<2>();
  ASSERT_NEAR(tri.area(), static_cast<F>(0.5), eps);
  tri[3] = um2::Point2(static_cast<F>(0.5), static_cast<F>(0.05));
  tri[5] = um2::Point2(static_cast<F>(0.05), static_cast<F>(0.5));
  // Actually making this a static assert causes a compiler error.
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(tri.area(), static_cast<F>(0.4333333333), eps);

  um2::QuadraticTriangle<2> const tri2 = makeTri2<2>();
  ASSERT_NEAR(tri2.area(), static_cast<F>(0.83333333), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// centroid
//==============================================================================

HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticTriangle<2> const tri = makeTri<2>();
  um2::Point<2> c = tri.centroid();
  ASSERT_NEAR(c[0], static_cast<F>(1.0 / 3.0), eps);
  ASSERT_NEAR(c[1], static_cast<F>(1.0 / 3.0), eps);

  um2::QuadraticTriangle<2> const tri2 = makeTri2<2>();
  c = tri2.centroid();
  ASSERT_NEAR(c[0], static_cast<F>(0.432), eps);
  ASSERT_NEAR(c[1], static_cast<F>(0.448), eps);
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
  ASSERT_NEAR(box.minima()[0], static_cast<F>(0), eps);
  ASSERT_NEAR(box.minima()[1], static_cast<F>(0), eps);
  ASSERT_NEAR(box.maxima()[0], static_cast<F>(1), eps);
  ASSERT_NEAR(box.maxima()[1], static_cast<F>(1.008333), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  auto tri = makeTri<2>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  um2::swap(tri[3], tri[5]);
  ASSERT(!tri.isCCW());
  um2::flipFace(tri);
  ASSERT(tri.isCCW());
}

//==============================================================================
// meanChordLength
//==============================================================================

HOSTDEV
TEST_CASE(meanChordLength)
{
  auto const tri = makeTri<2>();
  auto const two = static_cast<F>(2);
  auto const ref = um2::pi<F> / (two * (two + um2::sqrt(two)));
  ASSERT_NEAR(tri.meanChordLength(), ref, static_cast<F>(1e-4));
}

#if UM2_USE_CUDA
template <Size D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Size D>
MAKE_CUDA_KERNEL(jacobian, D);

template <Size D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(contains);

MAKE_CUDA_KERNEL(area);

MAKE_CUDA_KERNEL(centroid);

MAKE_CUDA_KERNEL(boundingBox);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif // UM2_USE_CUDA

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

template <Size D>
TEST_SUITE(QuadraticTriangle)
{
  TEST_HOSTDEV(interpolate, 1, 1, D);
  TEST_HOSTDEV(jacobian, 1, 1, D);
  TEST_HOSTDEV(edge, 1, 1, D);
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
  RUN_SUITE(QuadraticTriangle<2>);
  RUN_SUITE(QuadraticTriangle<3>);
  return 0;
}
