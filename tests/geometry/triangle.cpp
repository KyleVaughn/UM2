#include "../test_framework.hpp"
#include <um2/geometry/triangle.hpp>

template <len_t D, typename T>
UM2_HOSTDEV static constexpr auto makeTri() -> um2::Triangle<D, T>
{
  um2::Triangle<D, T> this_tri;
  for (len_t i = 0; i < 3; ++i) {
    for (len_t j = 0; j < D; ++j) {
      this_tri[i][j] = static_cast<T>(0);
    }
  }
  this_tri[1][0] = static_cast<T>(1);
  this_tri[2][1] = static_cast<T>(1);
  return this_tri;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(interpolate)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Point<D, T> p00 = tri(0, 0);
  um2::Point<D, T> p10 = tri(1, 0);
  um2::Point<D, T> p01 = tri(0, 1);
  EXPECT_TRUE(um2::isApprox(p00, tri[0]));
  EXPECT_TRUE(um2::isApprox(p10, tri[1]));
  EXPECT_TRUE(um2::isApprox(p01, tri[2]));
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(jacobian)
{
  // For the reference triangle, the Jacobian is constant.
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Mat<D, 2, T> jac = tri.jacobian(0, 0);
  EXPECT_NEAR((jac(0, 0)), 1, 1e-5);
  EXPECT_NEAR((jac(1, 0)), 0, 1e-5);
  EXPECT_NEAR((jac(0, 1)), 0, 1e-5);
  EXPECT_NEAR((jac(1, 1)), 1, 1e-5);
  jac = tri.jacobian(static_cast<T>(0.2), static_cast<T>(0.3));
  EXPECT_NEAR((jac(0, 0)), 1, 1e-5);
  EXPECT_NEAR((jac(1, 0)), 0, 1e-5);
  EXPECT_NEAR((jac(0, 1)), 0, 1e-5);
  EXPECT_NEAR((jac(1, 1)), 1, 1e-5);
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(edge)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::LineSegment<D, T> edge = tri.edge(0);
  EXPECT_TRUE(um2::isApprox(edge[0], tri[0]));
  EXPECT_TRUE(um2::isApprox(edge[1], tri[1]));
  edge = tri.edge(1);
  EXPECT_TRUE(um2::isApprox(edge[0], tri[1]));
  EXPECT_TRUE(um2::isApprox(edge[1], tri[2]));
  edge = tri.edge(2);
  EXPECT_TRUE(um2::isApprox(edge[0], tri[2]));
  EXPECT_TRUE(um2::isApprox(edge[1], tri[0]));
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV TEST_CASE(contains)
{
  um2::Triangle<2, T> tri = makeTri<2, T>();
  um2::Point2<T> p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(0.25));
  EXPECT_TRUE(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.25));
  EXPECT_TRUE(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(1.25), static_cast<T>(0.25));
  EXPECT_FALSE(tri.contains(p));
  p = um2::Point2<T>(static_cast<T>(0.25), static_cast<T>(-0.25));
  EXPECT_FALSE(tri.contains(p));
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(area)
{
  um2::Triangle<2, T> tri = makeTri<2, T>();
  EXPECT_NEAR(area(tri), 0.5, 1e-5);
}
// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(centroid)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::Point<D, T> c = centroid(tri);
  EXPECT_NEAR(c[0], 1.0 / 3.0, 1e-5);
  EXPECT_NEAR(c[1], 1.0 / 3.0, 1e-5);
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV TEST_CASE(boundingBox)
{
  um2::Triangle<D, T> tri = makeTri<D, T>();
  um2::AABox<D, T> box = boundingBox(tri);
  EXPECT_NEAR(box.xmin(), 0, 1e-5);
  EXPECT_NEAR(box.ymin(), 0, 1e-5);
  EXPECT_NEAR(box.xmax(), 1, 1e-5);
  EXPECT_NEAR(box.ymax(), 1, 1e-5);
}

#if UM2_ENABLE_CUDA
template <len_t D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(edge, D, T);

template <typename T>
MAKE_CUDA_KERNEL(contains, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(area, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(centroid, D, T);

template <len_t D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T);
#endif // UM2_HAS_CUDA

template <len_t D, typename T>
TEST_SUITE(triangle)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(edge, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(contains, 1, 1, T);
  }
  TEST_HOSTDEV(area, 1, 1, D, T);
  TEST_HOSTDEV(centroid, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
}

auto main() -> int
{
  RUN_TESTS((triangle<2, float>));
  RUN_TESTS((triangle<3, float>));
  RUN_TESTS((triangle<2, double>));
  RUN_TESTS((triangle<3, double>));
  return 0;
}
