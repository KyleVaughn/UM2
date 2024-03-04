#include <um2/geometry/polygon.hpp>

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
  auto const two = castIfNot<Float>(2);
  auto const ref = um2::pi<Float> / (two * (two + um2::sqrt(two)));

  std::cerr << "tri.meanChordLength() = " << tri.meanChordLength() << std::endl;
  std::cerr << "ref = " << ref << std::endl;
  ASSERT_NEAR(tri.meanChordLength(), ref, castIfNot<Float>(1e-4));
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
