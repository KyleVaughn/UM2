#include <um2/geometry/QuadraticTriangle.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeTri() -> um2::QuadraticTriangle<D, T>
{
  um2::QuadraticTriangle<D, T> this_tri;
  for (Size i = 0; i < 6; ++i) {
    this_tri[i] = um2::zeroVec<D, T>();
  }
  this_tri[1][0] = static_cast<T>(1);
  this_tri[2][1] = static_cast<T>(1);
  this_tri[3][0] = static_cast<T>(0.5);
  this_tri[4][0] = static_cast<T>(0.5);
  this_tri[4][1] = static_cast<T>(0.5);
  this_tri[5][1] = static_cast<T>(0.5);
  return this_tri;
}

// P4 = (0.7, 0.8)
template <Size D, typename T>
HOSTDEV static constexpr auto
makeTri2() -> um2::QuadraticTriangle<D, T>
{
  um2::QuadraticTriangle<D, T> this_tri;
  for (Size i = 0; i < 6; ++i) {
    this_tri[i] = um2::zeroVec<D, T>();
  }
  this_tri[1][0] = static_cast<T>(1);
  this_tri[2][1] = static_cast<T>(1);
  this_tri[3][0] = static_cast<T>(0.5);
  this_tri[4][0] = static_cast<T>(0.7);
  this_tri[4][1] = static_cast<T>(0.8);
  this_tri[5][1] = static_cast<T>(0.5);
  return this_tri;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticTriangle<D, T> tri = makeTri2<D, T>();
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

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::QuadraticTriangle<D, T> tri = makeTri<D, T>();
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
  um2::QuadraticTriangle<D, T> tri = makeTri2<D, T>();
  static_assert(numEdges(tri) == 3);
  um2::QuadraticSegment<D, T> edge = tri.getEdge(0);
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

template <typename T>
HOSTDEV
TEST_CASE(contains)
{
  um2::QuadraticTriangle<2, T> const tri = makeTri2<2, T>();
  um2::Point2<T> p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(1.25), static_cast<T>(0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(-0.25));
  ASSERT(!tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.6), static_cast<T>(0.6));
  ASSERT(tri.contains(p));
}

//==============================================================================
// area
//==============================================================================

template <typename T>
HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticTriangle<2, T> tri = makeTri<2, T>();
  ASSERT_NEAR(tri.area(), static_cast<T>(0.5), static_cast<T>(1e-5));
  tri[3] = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.05));
  tri[5] = um2::Point2<T>(static_cast<T>(0.05), static_cast<T>(0.5));
  ASSERT_NEAR(tri.area(), static_cast<T>(0.4333333333), static_cast<T>(1e-5));

  um2::QuadraticTriangle<2, T> const tri2 = makeTri2<2, T>();
  ASSERT_NEAR(tri2.area(), static_cast<T>(0.83333333), static_cast<T>(1e-5));
}

//==============================================================================
// centroid
//==============================================================================

template <typename T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticTriangle<2, T> const tri = makeTri<2, T>();
  um2::Point<2, T> c = tri.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(1.0 / 3.0), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(1.0 / 3.0), static_cast<T>(1e-5));

  um2::QuadraticTriangle<2, T> const tri2 = makeTri2<2, T>();
  c = tri2.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(0.432), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(0.448), static_cast<T>(1e-5));
}

//==============================================================================
// boundingBox
//==============================================================================

template <typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticTriangle<2, T> const tri = makeTri2<2, T>();
  um2::AxisAlignedBox<2, T> const box = tri.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1.008333), static_cast<T>(1e-5));
}

//==============================================================================
// isCCW
//==============================================================================

template <typename T>
HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  auto tri = makeTri<2, T>();
  ASSERT(tri.isCCW());
  um2::swap(tri[1], tri[2]);
  um2::swap(tri[3], tri[5]);
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

template <typename T>
MAKE_CUDA_KERNEL(area, T);

template <typename T>
MAKE_CUDA_KERNEL(centroid, T);

template <typename T>
MAKE_CUDA_KERNEL(boundingBox, T);

template <typename T>
MAKE_CUDA_KERNEL(isCCW_flipFace, T);
#endif // UM2_USE_CUDA

template <Size D, typename T>
TEST_SUITE(QuadraticTriangle)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(edge, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains, 1, 1, T);
    TEST_HOSTDEV(area, 1, 1, T);
    TEST_HOSTDEV(centroid, 1, 1, T);
    TEST_HOSTDEV(boundingBox, 1, 1, T);
    TEST_HOSTDEV(isCCW_flipFace, 1, 1, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((QuadraticTriangle<2, float>));
  RUN_SUITE((QuadraticTriangle<3, float>));
  RUN_SUITE((QuadraticTriangle<2, double>));
  RUN_SUITE((QuadraticTriangle<3, double>));
  return 0;
}
