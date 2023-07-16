#include <um2/geometry/QuadraticSegment.hpp>

#include "../test_macros.hpp"

// Description of the quadratic segments used in test cases
// --------------------------------------------------------
// All segment have P0 = (0, 0) and P1 = (2, 0)
// 1) A straight line segment with P2 = (1, 0) 
// 2) A segment that curves right with P2 = (1, 1)
// 3) A segment that curves left with P2 = (1, -1)
// 4) A segment that curves right, with P2 = (2, 1)
//     This segment has multiple r for the same x value
//     x_max = 2.25, y_max = 1
// 5) A segment that curves left, with P2 = (2, -1)
//     This segment has multiple r for the same x value
//     x_max = 2.25, y_min = -1
// 6) A segment that curves right, with P2 = (0, 1)
//     This segment has multiple r for the same x value
//     x_min = -0.25, y_max = 1
// 7) A segment that curves left, with P2 = (0, -1)
//    This segment has multiple r for the same x value
//    x_min = -0.25, y_min = -1
// 8) A segment (0,0) -> (2, 0) -> (4, 3)

template <Size D, typename T>
HOSTDEV static constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q; 
  q[0] = um2::zeroVec<D, T>();
  q[1] = um2::zeroVec<D, T>();
  q[2] = um2::zeroVec<D, T>();
  q[1][0] = static_cast<T>(2);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg1() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg2() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(1);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg3() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(1);
  q[2][1] = static_cast<T>(-1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg4() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg5() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(2);
  q[2][1] = static_cast<T>(-1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg6() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(0);
  q[2][1] = static_cast<T>(1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg7() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = static_cast<T>(0);
  q[2][1] = static_cast<T>(-1);
  return q;
}

template <Size D, typename T>
HOSTDEV static constexpr auto
makeSeg8() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[1][0] = static_cast<T>(2);
  q[2][0] = static_cast<T>(4);
  q[2][1] = static_cast<T>(3);
  return q;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticSegment<D, T> seg = makeSeg2<D, T>();
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
  um2::QuadraticSegment<D, T> seg = makeSeg1<D, T>();
  um2::Vec<D, T> j0 = seg.jacobian(0);
  um2::Vec<D, T> j12 = seg.jacobian(static_cast<T>(0.5));
  um2::Vec<D, T> j1 = seg.jacobian(1);
  um2::Vec<D, T> j_ref = um2::zeroVec<D, T>();
  j_ref[0] = static_cast<T>(2);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j12, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));

  um2::QuadraticSegment<D, T> seg2 = makeSeg2<D, T>();
  j0 = seg2.jacobian(0);
  j12 = seg2.jacobian(static_cast<T>(0.5));
  j1 = seg2.jacobian(1);
  ASSERT_NEAR(j0[0], static_cast<T>(2), static_cast<T>(1e-5));
  ASSERT(j0[1] > 0);
  ASSERT_NEAR(j12[0], static_cast<T>(2), static_cast<T>(1e-5));
  ASSERT_NEAR(j12[1], static_cast<T>(0), static_cast<T>(1e-5));
  ASSERT_NEAR(j1[0], static_cast<T>(2), static_cast<T>(1e-5));
  ASSERT(j1[1] < 0);
}

// -------------------------------------------------------------------
// isStraight
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(isStraight)
{
  um2::QuadraticSegment<D, T> seg1 = makeSeg1<D, T>();
  ASSERT(seg1.isStraight());
  um2::QuadraticSegment<D, T> seg2 = makeSeg2<D, T>();
  ASSERT(!seg2.isStraight());
  um2::QuadraticSegment<D, T> seg5 = makeSeg5<D, T>();
  ASSERT(!seg5.isStraight());
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(length)
{
  um2::QuadraticSegment<D, T> seg = makeSeg1<D, T>();
  T l_ref = static_cast<T>(2);
  T l = seg.length();
  ASSERT_NEAR(l, l_ref, static_cast<T>(1e-5));

  seg[2][1] = static_cast<T>(1);
  l_ref = static_cast<T>(2 * 1.4789428575445974);
  l = seg.length();
  ASSERT_NEAR(l, l_ref, static_cast<T>(1e-5));
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticSegment<D, T> seg1 = makeSeg1<D, T>();
  um2::AxisAlignedBox<D, T> bb1 = seg1.boundingBox();
  um2::AxisAlignedBox<D, T> bb_ref(seg1[0], seg1[1]);
  ASSERT(um2::isApprox(bb1, bb_ref));

  um2::QuadraticSegment<D, T> seg2 = makeSeg2<D, T>();
  um2::AxisAlignedBox<D, T> bb2 = seg2.boundingBox();
  um2::AxisAlignedBox<D, T> bb_ref2(seg2[0], seg2[1]);
  bb_ref2.maxima.max(seg2[2]);
  ASSERT(um2::isApprox(bb2, bb_ref2));

  um2::QuadraticSegment<D, T> seg8 = makeSeg8<D, T>();
  um2::AxisAlignedBox<D, T> bb8 = seg8.boundingBox();
  um2::AxisAlignedBox<D, T> bb_ref8(um2::zeroVec<D, T>(), um2::zeroVec<D, T>());
  bb_ref8.maxima[0] = static_cast<T>(4.083334);
  bb_ref8.maxima[1] = static_cast<T>(3);
  ASSERT(um2::isApprox(bb8, bb_ref8));
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <typename T>
HOSTDEV
TEST_CASE(isLeft)
{
  um2::Vector<um2::Point2<T>> test_points =
  {
    um2::Point2<T>(static_cast<T>(1), static_cast<T>(3)), // always left
    um2::Point2<T>(static_cast<T>(1), static_cast<T>(-3)), // always right
    um2::Point2<T>(static_cast<T>(-1), static_cast<T>(0.5)), // always left
    um2::Point2<T>(static_cast<T>(-1), static_cast<T>(-0.5)), // always right
    um2::Point2<T>(static_cast<T>(3), static_cast<T>(0.5)), // always left
    um2::Point2<T>(static_cast<T>(3), static_cast<T>(-0.5)), // always right
    um2::Point2<T>(static_cast<T>(0.1), static_cast<T>(0.9)), // always left 
    um2::Point2<T>(static_cast<T>(0.1), static_cast<T>(-0.9)), // always right
    um2::Point2<T>(static_cast<T>(1.9), static_cast<T>(0.9)), // always left
    um2::Point2<T>(static_cast<T>(1.9), static_cast<T>(-0.9)), // always right
    um2::Point2<T>(static_cast<T>(1.1), static_cast<T>(0.5)), 
  };
  
  // A straight line
  um2::QuadraticSegment2<T> q1 = makeSeg1<2, T>();
  ASSERT(q1.isLeft(test_points[0]));
  ASSERT(!q1.isLeft(test_points[1]));
  ASSERT(q1.isLeft(test_points[2]));
  ASSERT(!q1.isLeft(test_points[3]));
  ASSERT(q1.isLeft(test_points[4]));
  ASSERT(!q1.isLeft(test_points[5]));
  ASSERT(q1.isLeft(test_points[6]));
  ASSERT(!q1.isLeft(test_points[7]));
  ASSERT(q1.isLeft(test_points[8]));
  ASSERT(!q1.isLeft(test_points[9]));
  ASSERT(q1.isLeft(test_points[10]));

  // Curves right
  um2::QuadraticSegment2<T> q2 = makeSeg2<2, T>();
  ASSERT(q2.isLeft(test_points[0]));
  ASSERT(!q2.isLeft(test_points[1]));
  ASSERT(q2.isLeft(test_points[2]));
  ASSERT(!q2.isLeft(test_points[3]));
  ASSERT(q2.isLeft(test_points[4]));
  ASSERT(!q2.isLeft(test_points[5]));
  ASSERT(q2.isLeft(test_points[6]));
  ASSERT(!q2.isLeft(test_points[7]));
  ASSERT(q2.isLeft(test_points[8]));
  ASSERT(!q2.isLeft(test_points[9]));
  ASSERT(!q2.isLeft(test_points[10]));

  // Curves left
  um2::QuadraticSegment2<T> q3 = makeSeg3<2, T>();
  ASSERT(q3.isLeft(test_points[0]));
  ASSERT(!q3.isLeft(test_points[1]));
  ASSERT(q3.isLeft(test_points[2]));
  ASSERT(!q3.isLeft(test_points[3]));
  ASSERT(q3.isLeft(test_points[4]));
  ASSERT(!q3.isLeft(test_points[5]));
  ASSERT(q3.isLeft(test_points[6]));
  ASSERT(!q3.isLeft(test_points[7]));
  ASSERT(q3.isLeft(test_points[8]));
  ASSERT(!q3.isLeft(test_points[9]));
  ASSERT(q3.isLeft(test_points[10]));
//  um2::Point2<T> p_down(static_cast<T>(1.1), static_cast<T>(-0.5));
//  ASSERT(!q1.isLeft(p_down)); 
//  ASSERT(!q2.isLeft(p_down));
//  ASSERT(q3.isLeft(p_down)); 
//
//  // A poorly behaved segment
//  um2::QuadraticSegment2<T> q4(p0, p1, um2::Point2<T>(2, 1));
//
}

#if UM2_ENABLE_CUDA
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
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
  TEST_HOSTDEV(jacobian, 1, 1, D, T);
  TEST_HOSTDEV(isStraight, 1, 1, D, T);
  TEST_HOSTDEV(boundingBox, 1, 1, D, T);
  TEST_HOSTDEV(length, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isLeft, 1, 1, T);
  }
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
