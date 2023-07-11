#include <um2/geometry/QuadraticSegment.hpp>

#include "../test_macros.hpp"

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticSegment<D, T> seg;
  seg[0] = um2::zeroVec<D, T>();
  seg[1] = um2::zeroVec<D, T>();
  seg[2] = um2::zeroVec<D, T>();
  seg[1][0] = static_cast<T>(2);
  seg[2][0] = static_cast<T>(1);
  seg[2][1] = static_cast<T>(1);
  for (Size i = 0; i < 5; ++i) {
    T r = static_cast<T>(i) / static_cast<T>(4);
    um2::Point<D, T> p = seg(r);
    um2::Point<D, T> p_ref = um2::zeroVec<D, T>();
    p_ref[0] = 2 * r;
    p_ref[1] = 4 * r * (1 - r);
    ASSERT(um2::isApprox(p, p_ref));
  }
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::QuadraticSegment<D, T> seg;
  seg[0] = um2::zeroVec<D, T>();
  seg[1] = um2::zeroVec<D, T>();
  seg[2] = um2::zeroVec<D, T>();
  seg[1][0] = static_cast<T>(1);
  seg[2][0] = static_cast<T>(0.5);
  um2::Vec<D, T> j0 = seg.jacobian(0);
  um2::Vec<D, T> j12 = seg.jacobian(static_cast<T>(0.5));
  um2::Vec<D, T> j1 = seg.jacobian(1);
  um2::Vec<D, T> j_ref = um2::zeroVec<D, T>();
  j_ref[0] = static_cast<T>(1);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j12, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));
}

//// -------------------------------------------------------------------
//// length
//// -------------------------------------------------------------------
//
// template <Size D, typename T>
// HOSTDEV
// TEST_CASE(length)
//{
//  um2::QuadraticSegment<D, T> seg = makeQuadratic<D, T>();
//  T len_ref = seg[0].distanceTo(seg[1]);
//  ASSERT_NEAR(seg.length(), len_ref, static_cast<T>(1e-5));
//}
//
//// -------------------------------------------------------------------
//// boundingBox
//// -------------------------------------------------------------------
//
// template <Size D, typename T>
// HOSTDEV
// TEST_CASE(boundingBox)
//{
//  um2::QuadraticSegment<D, T> seg = makeQuadratic<D, T>();
//  um2::AxisAlignedBox<D, T> box = seg.boundingBox();
//  ASSERT(um2::isApprox(seg[0], box.minima));
//  ASSERT(um2::isApprox(seg[1], box.maxima));
//}
//
//// -------------------------------------------------------------------
//// isLeft
//// -------------------------------------------------------------------
//
// template <typename T>
// HOSTDEV
// TEST_CASE(isLeft)
//{
//  um2::QuadraticSegment2<T> seg = makeQuadratic<2, T>();
//  um2::Point2<T> p0 = seg[0];
//  um2::Point2<T> p1 = seg[1];
//  p0[1] -= static_cast<T>(1); // (1, 0)
//  p1[1] += static_cast<T>(1); // (2, 3)
//  ASSERT(!seg.isLeft(p0));
//  ASSERT(seg.isLeft(p1));
//  p0[1] += static_cast<T>(2); // (1, 2)
//  p1[1] -= static_cast<T>(2); // (2, 1)
//  ASSERT(seg.isLeft(p0));
//  ASSERT(!seg.isLeft(p1));
//  p1[1] = static_cast<T>(2.01); // (2, 2.01)
//  ASSERT(seg.isLeft(p1));
//}
//
#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(jacobian, D, T);

// template <Size D, typename T>
// MAKE_CUDA_KERNEL(length, D, T);
//
// template <Size D, typename T>
// MAKE_CUDA_KERNEL(boundingBox, D, T);
//
// template <typename T>
// MAKE_CUDA_KERNEL(isLeft, T);
#endif

template <Size D, typename T>
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  //  TEST_HOSTDEV(length, 1, 1, D, T);
  //  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  //  TEST_HOSTDEV(isLeft, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE((QuadraticSegment<2, float>));
  RUN_SUITE((QuadraticSegment<3, float>));
  RUN_SUITE((QuadraticSegment<2, double>));
  RUN_SUITE((QuadraticSegment<3, double>));
  return 0;
}
