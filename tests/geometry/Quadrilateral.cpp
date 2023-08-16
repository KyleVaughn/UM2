#include <um2/geometry/Quadrilateral.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeQuad() -> um2::Quadrilateral<D, T>
{
  um2::Quadrilateral<D, T> quad;
  for (Size i = 0; i < 4; ++i) {
    for (Size j = 0; j < D; ++j) {
      quad[i][j] = static_cast<T>(0);
    }
  }
  quad[1][0] = static_cast<T>(1);
  quad[2][0] = static_cast<T>(1);
  quad[2][1] = static_cast<T>(1);
  quad[3][1] = static_cast<T>(1);
  return quad;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeTriQuad() -> um2::Quadrilateral<D, T>
{
  um2::Quadrilateral<D, T> quad;
  for (Size i = 0; i < 4; ++i) {
    for (Size j = 0; j < D; ++j) {
      quad[i][j] = static_cast<T>(0);
    }
  }
  quad[1][0] = static_cast<T>(1);
  quad[2][0] = static_cast<T>(1);
  quad[2][1] = static_cast<T>(1);
  quad[3][1] = static_cast<T>(0.5);
  quad[3][0] = static_cast<T>(0.5);
  return quad;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  um2::Point<D, T> const p00 = quad(0, 0);
  um2::Point<D, T> const p10 = quad(1, 0);
  um2::Point<D, T> const p01 = quad(0, 1);
  um2::Point<D, T> const p11 = quad(1, 1);
  ASSERT(um2::isApprox(p00, quad[0]));
  ASSERT(um2::isApprox(p10, quad[1]));
  ASSERT(um2::isApprox(p01, quad[3]));
  ASSERT(um2::isApprox(p11, quad[2]));
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::Quadrilateral<D, T> const quad = makeQuad<D, T>();
  um2::Mat<D, 2, T> jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), static_cast<T>(1), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), static_cast<T>(1), static_cast<T>(1e-5));
  jac = quad.jacobian(static_cast<T>(0.2), static_cast<T>(0.3));
  ASSERT_NEAR((jac(0, 0)), static_cast<T>(1), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), static_cast<T>(1), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(edge)
{
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  um2::LineSegment<D, T> edge = quad.edge(0);
  ASSERT(um2::isApprox(edge[0], quad[0]));
  ASSERT(um2::isApprox(edge[1], quad[1]));
  edge = quad.edge(1);
  ASSERT(um2::isApprox(edge[0], quad[1]));
  ASSERT(um2::isApprox(edge[1], quad[2]));
  edge = quad.edge(2);
  ASSERT(um2::isApprox(edge[0], quad[2]));
  ASSERT(um2::isApprox(edge[1], quad[3]));
  edge = quad.edge(3);
  ASSERT(um2::isApprox(edge[0], quad[3]));
  ASSERT(um2::isApprox(edge[1], quad[0]));
}

// -------------------------------------------------------------------
// isConvex
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(isConvex)
{
  um2::Quadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT(quad.isConvex());
  quad[3][0] = static_cast<T>(0.5);
  ASSERT(quad.isConvex());
  quad[3][1] = static_cast<T>(0.5);
  ASSERT(quad.isConvex()); // Effectively a triangle.
  quad[3][0] = static_cast<T>(0.75);
  ASSERT(!quad.isConvex());
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(contains)
{
  um2::Quadrilateral<2, T> const quad = makeQuad<2, T>();
  um2::Point2<T> p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(1.25), static_cast<T>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(-0.25));
  ASSERT(!quad.contains(p));
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(area)
{
  um2::Quadrilateral<2, T> const quad = makeQuad<2, T>();
  ASSERT_NEAR(quad.area(), static_cast<T>(1), static_cast<T>(1e-5));
  um2::Quadrilateral<2, T> const triquad = makeTriQuad<2, T>();
  ASSERT_NEAR(triquad.area(), static_cast<T>(0.5), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::Quadrilateral<D, T> quad = makeQuad<D, T>();
  um2::Point<D, T> c = quad.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(0.5), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(0.5), static_cast<T>(1e-5));
  quad[2] = um2::Point<D, T>(static_cast<T>(2), static_cast<T>(0.5));
  quad[3] = um2::Point<D, T>(static_cast<T>(1), static_cast<T>(0.5));
  c = quad.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(1.00), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(0.25), static_cast<T>(1e-5));
  um2::Quadrilateral<D, T> const quad2 = makeTriQuad<D, T>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(static_cast<T>(2) / 3), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(static_cast<T>(1) / 3), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::Quadrilateral<D, T> const quad = makeQuad<D, T>();
  um2::AxisAlignedBox<D, T> const box = quad.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// isCCW
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(isCCW_flipFace)
{
  um2::Quadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT(quad.isCCW());
  um2::swap(quad[1], quad[3]);
  ASSERT(!quad.isCCW());
  um2::flipFace(quad);
  ASSERT(quad.isCCW());
}

#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(edge, D, T);

template <typename T>
MAKE_CUDA_KERNEL(isConvex, T);

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
TEST_SUITE(Quadrilateral)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(edge, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isConvex, 1, 1, T);
    TEST_HOSTDEV(contains, 1, 1, T);
    TEST_HOSTDEV(isCCW_flipFace, 1, 1, T);
  }
  TEST_HOSTDEV(area, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(centroid, 1, 1, D, T);
  }
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
}

auto
main() -> int
{
  RUN_SUITE((Quadrilateral<2, float>));
  RUN_SUITE((Quadrilateral<3, float>));
  RUN_SUITE((Quadrilateral<2, double>));
  RUN_SUITE((Quadrilateral<3, double>));
  return 0;
}
