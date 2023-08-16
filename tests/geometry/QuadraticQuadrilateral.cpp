#include <um2/geometry/QuadraticQuadrilateral.hpp>

#include "../test_macros.hpp"

#include <iostream>

template <Size D, typename T>
HOSTDEV static constexpr auto
makeQuad() -> um2::QuadraticQuadrilateral<D, T>
{
  um2::QuadraticQuadrilateral<D, T> this_quad;
  for (Size i = 0; i < 8; ++i) {
    this_quad[i] = um2::zeroVec<D, T>();
  }
  this_quad[1][0] = static_cast<T>(1);
  this_quad[2][0] = static_cast<T>(1);
  this_quad[2][1] = static_cast<T>(1);
  this_quad[3][1] = static_cast<T>(1);
  this_quad[4][0] = static_cast<T>(0.5);
  this_quad[5][0] = static_cast<T>(1);
  this_quad[5][1] = static_cast<T>(0.5);
  this_quad[6][0] = static_cast<T>(0.5);
  this_quad[6][1] = static_cast<T>(1);
  this_quad[7][1] = static_cast<T>(0.5);
  return this_quad;
}

// P6 = (0.8, 1.5)
template <Size D, typename T>
HOSTDEV static constexpr auto
makeQuad2() -> um2::QuadraticQuadrilateral<D, T>
{
  um2::QuadraticQuadrilateral<D, T> this_quad = makeQuad<D, T>();
  this_quad[6][0] = static_cast<T>(0.8);
  this_quad[6][1] = static_cast<T>(1.5);
  return this_quad;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad2<D, T>();
  ASSERT(um2::isApprox(quad(0, 0), quad[0]));
  ASSERT(um2::isApprox(quad(1, 0), quad[1]));
  ASSERT(um2::isApprox(quad(1, 1), quad[2]));
  ASSERT(um2::isApprox(quad(0, 1), quad[3]));
  ASSERT(um2::isApprox(quad(0.5, 0), quad[4]));
  ASSERT(um2::isApprox(quad(1, 0.5), quad[5]));
  ASSERT(um2::isApprox(quad(0.5, 1), quad[6]));
  ASSERT(um2::isApprox(quad(0, 0.5), quad[7]));
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  // For the reference quad, the Jacobian is constant.
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad<D, T>();
  um2::Mat<D, 2, T> jac = quad.jacobian(0, 0);
  ASSERT_NEAR((jac(0, 0)), 1, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), 1, static_cast<T>(1e-5));
  jac = quad.jacobian(static_cast<T>(0.2), static_cast<T>(0.3));
  ASSERT_NEAR((jac(0, 0)), 1, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 0)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(0, 1)), 0, static_cast<T>(1e-5));
  ASSERT_NEAR((jac(1, 1)), 1, static_cast<T>(1e-5));
  // If we stretch the quad, the Jacobian should change.
  quad[1][0] = static_cast<T>(2);
  jac = quad.jacobian(0.5, 0);
  ASSERT_NEAR((jac(0, 0)), 2, static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(edge)
{
  um2::QuadraticQuadrilateral<D, T> quad = makeQuad2<D, T>();
  um2::QuadraticSegment<D, T> edge = quad.edge(0);
  ASSERT(um2::isApprox(edge[0], quad[0]));
  ASSERT(um2::isApprox(edge[1], quad[1]));
  ASSERT(um2::isApprox(edge[2], quad[4]));
  edge = quad.edge(1);
  ASSERT(um2::isApprox(edge[0], quad[1]));
  ASSERT(um2::isApprox(edge[1], quad[2]));
  ASSERT(um2::isApprox(edge[2], quad[5]));
  edge = quad.edge(2);
  ASSERT(um2::isApprox(edge[0], quad[2]));
  ASSERT(um2::isApprox(edge[1], quad[3]));
  ASSERT(um2::isApprox(edge[2], quad[6]));
  edge = quad.edge(3);
  ASSERT(um2::isApprox(edge[0], quad[3]));
  ASSERT(um2::isApprox(edge[1], quad[0]));
  ASSERT(um2::isApprox(edge[2], quad[7]));
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(contains)
{
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad2<2, T>();
  um2::Point2<T> p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(2.25), static_cast<T>(0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(-0.25));
  ASSERT(!quad.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.8), static_cast<T>(1.3));
  ASSERT(quad.contains(p));
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(area)
{
  um2::QuadraticQuadrilateral<2, T> quad = makeQuad<2, T>();
  ASSERT_NEAR(quad.area(), static_cast<T>(1), static_cast<T>(1e-5));
  quad[5] = um2::Point2<T>(static_cast<T>(1.1), static_cast<T>(0.5));
  quad[7] = um2::Point2<T>(static_cast<T>(0.1), static_cast<T>(0.5));
  ASSERT_NEAR(quad.area(), static_cast<T>(1), static_cast<T>(1e-5));

  um2::QuadraticQuadrilateral<2, T> const quad2 = makeQuad2<2, T>();
  ASSERT_NEAR(quad2.area(), static_cast<T>(1.3333333), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad<2, T>();
  um2::Point<2, T> c = quad.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(0.5), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(0.5), static_cast<T>(1e-5));

  um2::QuadraticQuadrilateral<2, T> const quad2 = makeQuad2<2, T>();
  c = quad2.centroid();
  ASSERT_NEAR(c[0], static_cast<T>(0.53), static_cast<T>(1e-5));
  ASSERT_NEAR(c[1], static_cast<T>(0.675), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticQuadrilateral<2, T> const quad = makeQuad2<2, T>();
  um2::AxisAlignedBox<2, T> const box = quad.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1.0083333), static_cast<T>(1e-5));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1.5), static_cast<T>(1e-5));
}

// -------------------------------------------------------------------        
// isCCW        
// -------------------------------------------------------------------        
        
template <typename T>        
HOSTDEV        
TEST_CASE(isCCW_flipFace)        
{        
  auto quad = makeQuad<2, T>();    
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
MAKE_CUDA_KERNEL(contains, T);

template <typename T>
MAKE_CUDA_KERNEL(area, T);

template <typename T>
MAKE_CUDA_KERNEL(centroid, T);

template <typename T>
MAKE_CUDA_KERNEL(boundingBox, T);

template <typename T>
MAKE_CUDA_KERNEL(isCCW_flipFace, T);
#endif // UM2_ENABLE

template <Size D, typename T>
TEST_SUITE(QuadraticQuadrilateral)
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
  RUN_SUITE((QuadraticQuadrilateral<2, float>));
  RUN_SUITE((QuadraticQuadrilateral<3, float>));
  RUN_SUITE((QuadraticQuadrilateral<2, double>));
  RUN_SUITE((QuadraticQuadrilateral<3, double>));
  return 0;
}
