#include <um2/geometry/dion.hpp>

#include "../../test_macros.hpp"

// Ignore useless casts on initialization of points
// Point(static_cast<D>(0.1), static_cast<F>(0.2)) is not worth addressing
#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

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

F constexpr eps = um2::eps_distance * static_cast<F>(10);
F constexpr half = static_cast<F>(1) / static_cast<F>(2);

template <Size D>
HOSTDEV constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q;
  q[0] = um2::Vec<D, F>::zero();
  q[1] = um2::Vec<D, F>::zero();
  q[2] = um2::Vec<D, F>::zero();
  q[1][0] = static_cast<F>(2);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg1() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg2() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(1);
  q[2][1] = static_cast<F>(1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg3() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(1);
  q[2][1] = static_cast<F>(-1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg4() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(2);
  q[2][1] = static_cast<F>(1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg5() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(2);
  q[2][1] = static_cast<F>(-1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg6() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(0);
  q[2][1] = static_cast<F>(1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg7() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = static_cast<F>(0);
  q[2][1] = static_cast<F>(-1);
  return q;
}

template <Size D>
HOSTDEV constexpr auto
makeSeg8() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[1][0] = static_cast<F>(2);
  q[2][0] = static_cast<F>(4);
  q[2][1] = static_cast<F>(3);
  return q;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(interpolate)
{

  um2::QuadraticSegment<D> const seg = makeSeg2<D>();
  for (Size i = 0; i < 5; ++i) {
    F const r = static_cast<F>(i) / static_cast<F>(4);
    um2::Point<D> const p = seg(r);
    um2::Point<D> p_ref = um2::Vec<D, F>::zero();
    p_ref[0] = 2 * r;
    p_ref[1] = 4 * r * (1 - r);
    ASSERT(um2::isApprox(p, p_ref));
  }
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::QuadraticSegment<D> const seg = makeSeg1<D>();
  um2::Vec<D, F> j0 = seg.jacobian(0);
  um2::Vec<D, F> j12 = seg.jacobian(half);
  um2::Vec<D, F> j1 = seg.jacobian(1);
  um2::Vec<D, F> j_ref = um2::Vec<D, F>::zero();
  j_ref[0] = static_cast<F>(2);
  ASSERT(um2::isApprox(j0, j_ref));
  ASSERT(um2::isApprox(j12, j_ref));
  ASSERT(um2::isApprox(j1, j_ref));

  um2::QuadraticSegment<D> const seg2 = makeSeg2<D>();
  j0 = seg2.jacobian(0);
  j12 = seg2.jacobian(half);
  j1 = seg2.jacobian(1);
  ASSERT_NEAR(j0[0], static_cast<F>(2), eps);
  ASSERT(j0[1] > 0);
  ASSERT_NEAR(j12[0], static_cast<F>(2), eps);
  ASSERT_NEAR(j12[1], static_cast<F>(0), eps);
  ASSERT_NEAR(j1[0], static_cast<F>(2), eps);
  ASSERT(j1[1] < 0);
}

//==============================================================================
// length
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(length)
{
  um2::QuadraticSegment<D> seg = makeSeg1<D>();
  F l_ref = static_cast<F>(2);
  F l = seg.length();
  ASSERT_NEAR(l, l_ref, eps);

  seg[2][1] = static_cast<F>(1);
  // sqrt(5) + log(2 + sqrt(5)) / 2
  l_ref = static_cast<F>(2.957885715089195);
  l = seg.length();
  ASSERT_NEAR(l, l_ref, eps);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticSegment<D> seg1 = makeSeg1<D>();
  um2::AxisAlignedBox<D> const bb1 = seg1.boundingBox();
  um2::AxisAlignedBox<D> const bb_ref(seg1[0], seg1[1]);
  ASSERT(um2::isApprox(bb1, bb_ref));

  um2::QuadraticSegment<D> seg2 = makeSeg2<D>();
  um2::AxisAlignedBox<D> const bb2 = seg2.boundingBox();
  um2::AxisAlignedBox<D> bb_ref2(seg2[0], seg2[1]);
  bb_ref2 += seg2[2];
  ASSERT(um2::isApprox(bb2, bb_ref2));

  um2::QuadraticSegment<D> const seg8 = makeSeg8<D>();
  um2::AxisAlignedBox<D> const bb8 = seg8.boundingBox();
  um2::Point<D> const p0 = um2::Vec<D, F>::zero();
  um2::Point<D> p1 = um2::Vec<D, F>::zero();
  p1[0] = static_cast<F>(4.083334);
  p1[1] = static_cast<F>(3);
  um2::AxisAlignedBox<D> const bb_ref8(p0, p1);
  ASSERT(um2::isApprox(bb8, bb_ref8));
}

//==============================================================================
// isLeft
//==============================================================================

HOSTDEV
TEST_CASE(isLeft)
{
  um2::Vector<um2::Point2> const test_points = {
      um2::Point2(static_cast<F>(1), static_cast<F>(3)),      // always left
      um2::Point2(static_cast<F>(1), static_cast<F>(-3)),     // always right
      um2::Point2(static_cast<F>(-1), half),   // always left
      um2::Point2(static_cast<F>(-1), static_cast<F>(-0.5)),  // always right
      um2::Point2(static_cast<F>(3), half),    // always left
      um2::Point2(static_cast<F>(3), static_cast<F>(-0.5)),   // always right
      um2::Point2(static_cast<F>(0.1), static_cast<F>(0.9)),  // always left
      um2::Point2(static_cast<F>(0.1), static_cast<F>(-0.9)), // always right
      um2::Point2(static_cast<F>(1.9), static_cast<F>(0.9)),  // always left
      um2::Point2(static_cast<F>(1.9), static_cast<F>(-0.9)), // always right
      um2::Point2(static_cast<F>(1.1), half),
      um2::Point2(static_cast<F>(2), half),
      um2::Point2(static_cast<F>(2.1), static_cast<F>(0.01)),
      um2::Point2(static_cast<F>(2.1), half),
  };

  // A straight line
  um2::QuadraticSegment2 const q1 = makeSeg1<2>();
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
  ASSERT(q1.isLeft(test_points[11]));
  ASSERT(q1.isLeft(test_points[12]));
  ASSERT(q1.isLeft(test_points[13]));

  // Curves right
  um2::QuadraticSegment2 const q2 = makeSeg2<2>();
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
  ASSERT(q2.isLeft(test_points[11]));
  ASSERT(q2.isLeft(test_points[12]));
  ASSERT(q2.isLeft(test_points[13]));

  // Curves left
  um2::QuadraticSegment2 const q3 = makeSeg3<2>();
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
  ASSERT(q3.isLeft(test_points[11]));
  ASSERT(q3.isLeft(test_points[12]));
  ASSERT(q3.isLeft(test_points[13]));

  // Curves right, P2 = (2, 1)
  um2::QuadraticSegment2 const q4 = makeSeg4<2>();
  ASSERT(q4.isLeft(test_points[0]));
  ASSERT(!q4.isLeft(test_points[1]));
  ASSERT(q4.isLeft(test_points[2]));
  ASSERT(!q4.isLeft(test_points[3]));
  ASSERT(q4.isLeft(test_points[4]));
  ASSERT(!q4.isLeft(test_points[5]));
  ASSERT(q4.isLeft(test_points[6]));
  ASSERT(!q4.isLeft(test_points[7]));
  ASSERT(!q4.isLeft(test_points[8]));
  ASSERT(!q4.isLeft(test_points[9]));
  ASSERT(!q4.isLeft(test_points[10]));
  ASSERT(!q4.isLeft(test_points[11]));
  ASSERT(q4.isLeft(test_points[12]));
  ASSERT(!q4.isLeft(test_points[13]));

  // Curves left, P2 = (2, -1)
  um2::QuadraticSegment2 const q5 = makeSeg5<2>();
  ASSERT(q5.isLeft(test_points[0]));
  ASSERT(!q5.isLeft(test_points[1]));
  ASSERT(q5.isLeft(test_points[2]));
  ASSERT(!q5.isLeft(test_points[3]));
  ASSERT(q5.isLeft(test_points[4]));
  ASSERT(!q5.isLeft(test_points[5]));
  ASSERT(q5.isLeft(test_points[6]));
  ASSERT(!q5.isLeft(test_points[7]));
  ASSERT(q5.isLeft(test_points[8]));
  ASSERT(q5.isLeft(test_points[9]));
  ASSERT(q5.isLeft(test_points[10]));
  ASSERT(q5.isLeft(test_points[11]));
  ASSERT(q5.isLeft(test_points[12]));
  ASSERT(q5.isLeft(test_points[13]));

  // Curves right, P2 = (0, 1)
  um2::QuadraticSegment2 const q6 = makeSeg6<2>();
  ASSERT(q6.isLeft(test_points[0]));
  ASSERT(!q6.isLeft(test_points[1]));
  ASSERT(q6.isLeft(test_points[2]));
  ASSERT(!q6.isLeft(test_points[3]));
  ASSERT(q6.isLeft(test_points[4]));
  ASSERT(!q6.isLeft(test_points[5]));
  ASSERT(!q6.isLeft(test_points[6]));
  ASSERT(!q6.isLeft(test_points[7]));
  ASSERT(q6.isLeft(test_points[8]));
  ASSERT(!q6.isLeft(test_points[9]));
  ASSERT(!q6.isLeft(test_points[10]));
  ASSERT(q6.isLeft(test_points[11]));
  ASSERT(q6.isLeft(test_points[12]));
  ASSERT(q6.isLeft(test_points[13]));

  // Curves left, P2 = (0, -1)
  um2::QuadraticSegment2 const q7 = makeSeg7<2>();
  ASSERT(q7.isLeft(test_points[0]));
  ASSERT(!q7.isLeft(test_points[1]));
  ASSERT(q7.isLeft(test_points[2]));
  ASSERT(!q7.isLeft(test_points[3]));
  ASSERT(q7.isLeft(test_points[4]));
  ASSERT(!q7.isLeft(test_points[5]));
  ASSERT(q7.isLeft(test_points[6]));
  ASSERT(q7.isLeft(test_points[7]));
  ASSERT(q7.isLeft(test_points[8]));
  ASSERT(!q7.isLeft(test_points[9]));
  ASSERT(q7.isLeft(test_points[10]));
  ASSERT(q7.isLeft(test_points[11]));
  ASSERT(q7.isLeft(test_points[12]));
  ASSERT(q7.isLeft(test_points[13]));

  // Curves right, P2 = (4, 3)
  um2::QuadraticSegment2 const q8 = makeSeg8<2>();
  ASSERT(q8.isLeft(test_points[0]));
  ASSERT(!q8.isLeft(test_points[1]));
  ASSERT(q8.isLeft(test_points[2]));
  ASSERT(!q8.isLeft(test_points[3]));
  ASSERT(q8.isLeft(test_points[4]));
  ASSERT(!q8.isLeft(test_points[5]));
  ASSERT(q8.isLeft(test_points[6]));
  ASSERT(!q8.isLeft(test_points[7]));
  ASSERT(!q8.isLeft(test_points[8]));
  ASSERT(!q8.isLeft(test_points[9]));
  ASSERT(!q8.isLeft(test_points[10]));
  ASSERT(!q8.isLeft(test_points[11]));
  ASSERT(q8.isLeft(test_points[12]));
  ASSERT(!q8.isLeft(test_points[13]));
}

//==============================================================================
// pointClosestTo
//==============================================================================

HOSTDEV
TEST_CASE(pointClosestTo)
{
  // Due to difficulty in computing the closest point to a quadratic segment,
  // we will simply test a point, compute the distance, then perturb the value
  // in both directions and ensure that the distance increases.
  um2::Vector<um2::Point2> const test_points = {
      um2::Point2(static_cast<F>(1), static_cast<F>(3)),      // always left
      um2::Point2(static_cast<F>(1), static_cast<F>(-3)),     // always right
      um2::Point2(static_cast<F>(-1), half),   // always left
      um2::Point2(static_cast<F>(-1), static_cast<F>(-0.5)),  // always right
      um2::Point2(static_cast<F>(3), half),    // always left
      um2::Point2(static_cast<F>(3), static_cast<F>(-0.5)),   // always right
      um2::Point2(static_cast<F>(0.1), static_cast<F>(0.9)),  // always left
      um2::Point2(static_cast<F>(0.1), static_cast<F>(-0.9)), // always right
      um2::Point2(static_cast<F>(1.9), static_cast<F>(0.9)),  // always left
      um2::Point2(static_cast<F>(1.9), static_cast<F>(-0.9)), // always right
      um2::Point2(static_cast<F>(1.1), half),
      um2::Point2(static_cast<F>(2), half),
      um2::Point2(static_cast<F>(2.1), static_cast<F>(0.01)),
      um2::Point2(static_cast<F>(2.1), half),
  };
  F const big_eps = static_cast<F>(1e-2);
  um2::Vector<um2::QuadraticSegment2> const segments = {
      makeSeg1<2>(), makeSeg2<2>(), makeSeg3<2>(), makeSeg4<2>(),
      makeSeg5<2>(), makeSeg6<2>(), makeSeg7<2>(), makeSeg8<2>(),
  };

  for (auto const & q : segments) {
    for (um2::Point2 const & p : test_points) {
      F const r0 = q.pointClosestTo(p);
      F const d0 = p.distanceTo(q(r0));
      F const r1 = r0 - big_eps;
      F const d1 = p.distanceTo(q(r1));
      F const r2 = r0 + big_eps;
      F const d2 = p.distanceTo(q(r2));
      if (0 <= r1 && r1 <= 1) {
        ASSERT(d0 < d1);
      }
      if (0 <= r2 && r2 <= 1) {
        ASSERT(d0 < d2);
      }
    }
  }
}

//==============================================================================----------
// enclosedArea
//==============================================================================----------

HOSTDEV
TEST_CASE(enclosedArea)
{
  um2::QuadraticSegment2 const seg1 = makeSeg1<2>();
  F area = enclosedArea(seg1);
  F area_ref = static_cast<F>(0);
  ASSERT_NEAR(area, area_ref, eps);

  um2::QuadraticSegment2 const seg2 = makeSeg2<2>();
  // 4/3 triangle area = (2 / 3) * b * h
  area_ref = -static_cast<F>(2.0 / 3.0) * 2 * 1;
  area = enclosedArea(seg2);
  ASSERT_NEAR(area, area_ref, eps);

  um2::QuadraticSegment2 const seg4 = makeSeg4<2>();
  area = enclosedArea(seg4);
  ASSERT_NEAR(area, area_ref, eps);

  um2::QuadraticSegment2 const seg8 = makeSeg8<2>();
  area_ref = -static_cast<F>(4);
  area = enclosedArea(seg8);
  ASSERT_NEAR(area, area_ref, eps);
}

HOSTDEV
TEST_CASE(enclosedCentroid)
{
  um2::QuadraticSegment2 const seg1 = makeSeg1<2>();
  um2::Point2 centroid = enclosedCentroid(seg1);
  um2::Point2 centroid_ref(1, 0);
  ASSERT(um2::isApprox(centroid, centroid_ref));

  um2::QuadraticSegment2 seg2 = makeSeg2<2>();
  centroid_ref = um2::Point2(static_cast<F>(1), static_cast<F>(0.4));
  centroid = enclosedCentroid(seg2);
  ASSERT(um2::isApprox(centroid, centroid_ref));
  // Rotated 45 degrees, translated -1 in x
  seg2[0][0] = static_cast<F>(-1);
  seg2[1][0] = um2::sqrt(static_cast<F>(2)) - 1;
  seg2[1][1] = um2::sqrt(static_cast<F>(2));
  seg2[2][0] = static_cast<F>(-1);
  seg2[2][1] = um2::sqrt(static_cast<F>(2));
  // Compute centroid_ref
  um2::Vec2<F> const u1 = (seg2[1] - seg2[0]).normalized();
  um2::Vec2<F> const u2(-u1[1], u1[0]);
  // NOLINTNEXTLINE(readability-identifier-naming) justification: matrix notation
  um2::Mat2x2<F> const R(u1, u2);
  centroid_ref = R * centroid_ref + seg2[0];
  centroid = enclosedCentroid(seg2);
  ASSERT(um2::isApprox(centroid, centroid_ref));
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
MAKE_CUDA_KERNEL(length, D);

template <Size D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isLeft);

MAKE_CUDA_KERNEL(enclosedArea);

MAKE_CUDA_KERNEL(enclosedCentroid);

MAKE_CUDA_KERNEL(pointClosestTo);
#endif

template <Size D>
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, 1, 1, D);
  TEST_HOSTDEV(jacobian, 1, 1, D);
  TEST_HOSTDEV(boundingBox, 1, 1, D);
  TEST_HOSTDEV(length, 1, 1, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isLeft);
    TEST_HOSTDEV(pointClosestTo);
    TEST_HOSTDEV(enclosedArea);
    TEST_HOSTDEV(enclosedCentroid);
  }
}

auto
main() -> int
{
  RUN_SUITE(QuadraticSegment<2>);
  RUN_SUITE(QuadraticSegment<3>);
  return 0;
}
