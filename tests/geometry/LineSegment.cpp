#include <um2/geometry/LineSegment.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeLine() -> um2::LineSegment<D, T>
{
  um2::LineSegment<D, T> line;
  for (Size i = 0; i < D; ++i) {
    line.w[0][i] = static_cast<T>(1);
    line.w[1][i] = static_cast<T>(1);
  }
  return line;
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  ASSERT(um2::isApprox(line.getVertex(0), line.w[0]));
  um2::Vec<D, T> p1;
  for (Size i = 0; i < D; ++i) {
    p1[i] = line.w[0][i] + line.w[1][i];
  }
  ASSERT(um2::isApprox(line.getVertex(1), p1));
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Point<D, T> p0 = line(0);
  ASSERT((um2::isApprox(p0, line.w[0])));
  um2::Point<D, T> p1 = line(1);
  um2::Point<D, T> p05 = line(static_cast<T>(0.5));
  um2::Point<D, T> mp = um2::midpoint(p0, p1);
  ASSERT((um2::isApprox(p05, mp)));
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::Vec<D, T> j_ref = line.w[1];
  um2::Vec<D, T> j0 = line.jacobian(0);
  um2::Vec<D, T> j1 = line.jacobian(1);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(length)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  T len_ref = line.getVertex(0).distanceTo(line.getVertex(1));
  ASSERT_NEAR(line.length(), len_ref, static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::LineSegment<D, T> line = makeLine<D, T>();
  um2::AxisAlignedBox<D, T> box = line.boundingBox();
  ASSERT(um2::isApprox(line.getVertex(0), box.minima));
  ASSERT(um2::isApprox(line.getVertex(1), box.maxima));
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(isLeft)
{
  um2::LineSegment2<T> line = makeLine<2, T>();
  um2::Point2<T> p0 = line.getVertex(0);
  um2::Point2<T> p1 = line.getVertex(1);
  p0[1] -= static_cast<T>(1); // (1, 0)
  p1[1] += static_cast<T>(1); // (2, 3)
  ASSERT(!line.isLeft(p0));
  ASSERT(line.isLeft(p1));
  p0[1] += static_cast<T>(2); // (1, 2)
  p1[1] -= static_cast<T>(2); // (2, 1)
  ASSERT(line.isLeft(p0));
  ASSERT(!line.isLeft(p1));
  p1[1] = static_cast<T>(2.01); // (2, 2.01)
  ASSERT(line.isLeft(p1));
}

#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(length, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(boundingBox, D, T);

template <typename T>
MAKE_CUDA_KERNEL(isLeft, T);
#endif

template <Size D, typename T>
TEST_SUITE(LineSegment)
{
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(length, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  TEST_HOSTDEV(isLeft, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE((LineSegment<2, float>));
  RUN_SUITE((LineSegment<3, float>));
  RUN_SUITE((LineSegment<2, double>));
  RUN_SUITE((LineSegment<3, double>));
  return 0;
}
