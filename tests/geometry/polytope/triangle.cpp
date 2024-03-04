#include <um2/geometry/polygon.hpp>

#include "../../test_macros.hpp"

Float constexpr eps = um2::eps_distance * castIfNot<Float>(10);

template <Int D>
HOSTDEV constexpr auto
makeTri() -> um2::Triangle<D>
{
  um2::Triangle<D> this_tri;
  for (Int i = 0; i < 3; ++i) {
    for (Int j = 0; j < D; ++j) {
      this_tri[i][j] = castIfNot<Float>(0);
    }
  }
  this_tri[1][0] = castIfNot<Float>(1);
  this_tri[2][1] = castIfNot<Float>(1);
  return this_tri;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
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

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  // Floator the reference triangle, the Jacobian is constant.
  um2::Triangle<D> tri = makeTri<D>();
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
  um2::Point2 p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2(castIfNot<Float>(1.25), castIfNot<Float>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2(castIfNot<Float>(0.25), castIfNot<Float>(-0.25));
  ASSERT(!tri.contains(p));
}

//==============================================================================
// area
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(area)
{
  um2::Triangle<D> tri = makeTri<D>();
  ASSERT_NEAR(tri.area(), castIfNot<Float>(0.5), eps);
  tri[1][0] = castIfNot<Float>(2);
  ASSERT_NEAR(tri.area(), castIfNot<Float>(1), eps);
}

//==============================================================================
// perimeter
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(perimeter)
{
  um2::Triangle<D> const tri = makeTri<D>();
  auto const two = castIfNot<Float>(2);
  Float const ref = two + um2::sqrt(two);
  ASSERT_NEAR(tri.perimeter(), ref, eps);
}

//==============================================================================
// centroid
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::Point<D> c = tri.centroid();
  ASSERT_NEAR(c[0], castIfNot<Float>(1.0 / 3.0), eps);
  ASSERT_NEAR(c[1], castIfNot<Float>(1.0 / 3.0), eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Triangle<D> const tri = makeTri<D>();
  um2::AxisAlignedBox<D> const box = tri.boundingBox();
  ASSERT_NEAR(box.minima()[0], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima()[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima()[1], castIfNot<Float>(1), eps);
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
  auto const two = castIfNot<Float>(2);
  auto const ref = um2::pi<Float> / (two * (two + um2::sqrt(two)));
  ASSERT_NEAR(tri.meanChordLength(), ref, eps);
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Int D>
MAKE_CUDA_KERNEL(jacobian, D);

template <Int D>
MAKE_CUDA_KERNEL(edge, D);

MAKE_CUDA_KERNEL(contains);

template <Int D>
MAKE_CUDA_KERNEL(area, D);

template <Int D>
MAKE_CUDA_KERNEL(perimeter, D);

template <Int D>
MAKE_CUDA_KERNEL(centroid, D);

template <Int D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isCCW_flipFace);

MAKE_CUDA_KERNEL(meanChordLength);
#endif

template <Int D>
TEST_SUITE(Triangle)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(edge, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains);
    TEST_HOSTDEV(isCCW_flipFace);
    TEST_HOSTDEV(meanChordLength);
  }
  TEST_HOSTDEV(area, D);
  TEST_HOSTDEV(perimeter, D);
  TEST_HOSTDEV(centroid, D);
  TEST_HOSTDEV(boundingBox, D);
}

auto
main() -> int
{
  RUN_SUITE(Triangle<2>);
  RUN_SUITE(Triangle<3>);
  return 0;
}
