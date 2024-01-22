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
makeTri() -> um2::Triangle<D>
{
  um2::Triangle<D> this_tri;
  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < D; ++j) {
      this_tri[i][j] = static_cast<F>(0);
    }
  }
  this_tri[1][0] = static_cast<F>(1);
  this_tri[2][1] = static_cast<F>(1);
  return this_tri;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Triangle<D> tri = makeTri<D>();
  um2::Point<D> const p00 = tri(0, 0);
  um2::Point<D> const p10 = tri(1, 0);
  um2::Point<D> const p01 = tri(0, 1);
  ASSERT(um2::isApprox(p00, tri[0]));
  ASSERT(um2::isApprox(p10, tri[1]));
  ASSERT(um2::isApprox(p01, tri[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::Triangle<D> tri = makeTri<D>();
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
  um2::Triangle<D> tri = makeTri<D>();
  um2::LineSegment<D> edge = tri.getEdge(0);
  ASSERT(um2::isApprox(edge[0], tri[0]));
  ASSERT(um2::isApprox(edge[1], tri[1]));
  edge = tri.getEdge(1);
  ASSERT(um2::isApprox(edge[0], tri[1]));
  ASSERT(um2::isApprox(edge[1], tri[2]));
  edge = tri.getEdge(2);
  ASSERT(um2::isApprox(edge[0], tri[2]));
  ASSERT(um2::isApprox(edge[1], tri[0]));
}

//==============================================================================
// contains
//==============================================================================

HOSTDEV
TEST_CASE(contains)
{
  um2::Triangle<2> const tri = makeTri<2>();
  um2::Point2 p = um2::Point2(static_cast<F>(0.25), static_cast<F>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(static_cast<F>(0.5), static_cast<F>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(static_cast<F>(1.25), static_cast<F>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(static_cast<F>(0.25), static_cast<F>(-0.25));
  ASSERT(!tri.contains(p));
}

//==============================================================================
// area
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(area)
{
  um2::Triangle<D> tri = makeTri<D>();
  ASSERT_NEAR(tri.area(), static_cast<F>(0.5), eps);
  tri[1][0] = static_cast<F>(2);
  ASSERT_NEAR(tri.area(), static_cast<F>(1), eps);
}

//==============================================================================
// perimeter
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Triangle<D> const tri = makeTri<D>();
  F const two = static_cast<F>(2);
  F const ref = two + um2::sqrt(two);
  ASSERT_NEAR(tri.perimeter(), ref, eps);
}

//==============================================================================
// centroid
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::Point<D> c = tri.centroid();
  ASSERT_NEAR(c[0], static_cast<F>(1.0 / 3.0), eps);
  ASSERT_NEAR(c[1], static_cast<F>(1.0 / 3.0), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::AxisAlignedBox<D> const box = tri.boundingBox();
  ASSERT_NEAR(box.minima()[0], static_cast<F>(0), eps);
  ASSERT_NEAR(box.minima()[1], static_cast<F>(0), eps);
  ASSERT_NEAR(box.maxima()[0], static_cast<F>(1), eps);
  ASSERT_NEAR(box.maxima()[1], static_cast<F>(1), eps);
}

//==============================================================================
// isCCW
//==============================================================================

HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  um2::Triangle<2> tri = makeTri<2>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
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
  um2::Triangle<2> const tri = makeTri<2>();
  auto const two = static_cast<F>(2);
  auto const ref = um2::pi<F> / (two * (two + um2::sqrt(two)));
  ASSERT_NEAR(tri.meanChordLength(), ref, eps);
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

#if UM2_USE_CUDA
template <Size D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Size D>
MAKE_CUDA_KERNEL(jacobian, D);

template <Size D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(contains);

template <Size D>
MAKE_CUDA_KERNEL(area, D);

template <Size D>
MAKE_CUDA_KERNEL(perimeter, D);

template <Size D>
MAKE_CUDA_KERNEL(centroid, D);

template <Size D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif

template <Size D>
TEST_SUITE(Triangle)
{
  TEST_HOSTDEV(interpolate, 1, 1, D);
  TEST_HOSTDEV(jacobian, 1, 1, D);
  TEST_HOSTDEV(edge, 1, 1, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(isCCW_flipFace);
    TEST_HOSTDEV(meanChordLength);
  }
  TEST_HOSTDEV(area, 1, 1, D);
  TEST_HOSTDEV(perimeter, 1, 1, D);
  TEST_HOSTDEV(centroid, 1, 1, D);
  TEST_HOSTDEV(boundingBox, 1, 1, D);
}

auto
main() -> int
{
  RUN_SUITE(Triangle<2>);
  RUN_SUITE(Triangle<3>);
  return 0;
}
