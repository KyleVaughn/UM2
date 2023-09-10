#include <um2/geometry/Polygon.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV constexpr auto
makeTri() -> um2::Triangle<D, T>
{
  um2::Triangle<D, T> this_tri;
  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < D; ++j) {
      this_tri[i][j] = static_cast<T>(0);
    }
  }
  this_tri[1][0] = static_cast<T>(1);
  this_tri[2][1] = static_cast<T>(1);
  return this_tri;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Point<D, T> const p00 = tri(0, 0);
  um2::Point<D, T> const p10 = tri(1, 0);
  um2::Point<D, T> const p01 = tri(0, 1);
  ASSERT(um2::isApprox(p00, tri[0]));
  ASSERT(um2::isApprox(p10, tri[1]));
  ASSERT(um2::isApprox(p01, tri[2]));
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Mat<D, 2, T> jac = tri.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), 1, static_cast<T>(1e-5));
  jac = tri.jacobian(static_cast<T>(0.2), static_cast<T>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), 1, static_cast<T>(1e-5));
  // If we stretch the triangle, the Jacobian should change.
  tri[1][0] = static_cast<T>(2);
  jac = tri.jacobian(0.5, 0);
  ASSERT_NEAR((jac(0, 0)), 2, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), 1, static_cast<T>(1e-5));
}

//==============================================================================
// edge
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(edge)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::LineSegment<D, T> edge = tri.getEdge(0);
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

template <typename T>
HOSTDEV
TEST_CASE(contains)
{
  um2::Triangle<2, T> const tri = makeTri<2, T>();
  um2::Point2<T> p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(1.25), static_cast<T>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(-0.25));
  ASSERT(!tri.contains(p));
}

//==============================================================================
// area
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(area)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  ASSERT_NEAR(tri.area(), static_cast<T>(0.5), static_cast<T>(1e-5));
  tri[1][0] = static_cast<T>(2);
  ASSERT_NEAR(tri.area(), static_cast<T>(1), static_cast<T>(1e-5));
}

//==============================================================================
// centroid
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Triangle<D, T> const tri = makeTri<D, T>();
  um2::Point<D, T> c = tri.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(1.0 / 3.0), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(1.0 / 3.0), static_cast<T>(1e-5));
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Triangle<D, T> const tri = makeTri<D, T>();
  um2::AxisAlignedBox<D, T> const box = tri.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-5));
}

//==============================================================================
// isCCW
//==============================================================================

template <typename T>
HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  um2::Triangle<2, T> tri = makeTri<2, T>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  ASSERT(!tri.isCCW());
  um2::flipFace(tri);
  ASSERT(tri.isCCW());
}

#if UM2_USE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(edge, D, T);

template <typename T>
MAKE_CUDA_KERNEL(contains, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(area, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(centroid, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T);

template <typename T>
MAKE_CUDA_KERNEL(isCCW_flipFace, T);
#endif

template <Size D, typename T>
TEST_SUITE(Triangle)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(edge, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains, 1, 1, T);
    TEST_HOSTDEV(isCCW_flipFace, 1, 1, T);
  }
  TEST_HOSTDEV(area, 1, 1, D, T);
  TEST_HOSTDEV(centroid, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
}

auto
main() -> int
{
  RUN_SUITE((Triangle<2, float>));
  RUN_SUITE((Triangle<3, float>));
  RUN_SUITE((Triangle<2, double>));
  RUN_SUITE((Triangle<3, double>));
  return 0;
}
