#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/modular_rays.hpp>

#include <iostream>

#include "../../test_macros.hpp"

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

// CUDA is annoying and defines half, so we have to use ahalf
Float constexpr eps = um2::eps_distance * castIfNot<Float>(10);
Float constexpr ahalf = castIfNot<Float>(1) / castIfNot<Float>(2);

template <Int D>
HOSTDEV constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q;
  q[0] = um2::Vec<D, Float>::zero();
  q[1] = um2::Vec<D, Float>::zero();
  q[2] = um2::Vec<D, Float>::zero();
  q[1][0] = castIfNot<Float>(2);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg1() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg2() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(1);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg3() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(1);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg4() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(2);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg5() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(2);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg6() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(0);
  q[2][1] = castIfNot<Float>(1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg7() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[2][0] = castIfNot<Float>(0);
  q[2][1] = castIfNot<Float>(-1);
  return q;
}

template <Int D>
HOSTDEV constexpr auto
makeSeg8() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q = makeBaseSeg<D>();
  q[1][0] = castIfNot<Float>(2);
  q[2][0] = castIfNot<Float>(4);
  q[2][1] = castIfNot<Float>(3);
  return q;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{

  um2::QuadraticSegment<D> const seg = makeSeg2<D>();
  for (Int i = 0; i < 5; ++i) {
    Float const r = castIfNot<Float>(i) / castIfNot<Float>(4);
    um2::Point<D> const p = seg(r);
    um2::Point<D> p_ref = um2::Vec<D, Float>::zero();
    p_ref[0] = 2 * r;
    p_ref[1] = 4 * r * (1 - r);
    ASSERT(p.isApprox(p_ref));
  }
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(jacobian)
{
  um2::QuadraticSegment<D> const seg = makeSeg1<D>();
  um2::Vec<D, Float> j0 = seg.jacobian(0);
  um2::Vec<D, Float> j12 = seg.jacobian(ahalf);
  um2::Vec<D, Float> j1 = seg.jacobian(1);
  um2::Vec<D, Float> j_ref = um2::Vec<D, Float>::zero();
  j_ref[0] = castIfNot<Float>(2);
  ASSERT(j0.isApprox(j_ref));
  ASSERT(j12.isApprox(j_ref));
  ASSERT(j1.isApprox(j_ref));

  um2::QuadraticSegment<D> const seg2 = makeSeg2<D>();
  j0 = seg2.jacobian(0);
  j12 = seg2.jacobian(ahalf);
  j1 = seg2.jacobian(1);
  ASSERT_NEAR(j0[0], static_cast<Float>(2), eps);
  ASSERT(j0[1] > 0);
  ASSERT_NEAR(j12[0], castIfNot<Float>(2), eps);
  ASSERT_NEAR(j12[1], castIfNot<Float>(0), eps);
  ASSERT_NEAR(j1[0], castIfNot<Float>(2), eps);
  ASSERT(j1[1] < 0);
}

//==============================================================================
// length
//==============================================================================

template <Int D>
HOSTDEV
void
testLength(um2::QuadraticSegment<D> const & seg)
{
  // Note: this samples evenly in parametric space, not in physical space.
  Int constexpr num_segs = 10000;
  Float const dr = castIfNot<Float>(1) / castIfNot<Float>(num_segs);
  Float r = 0;
  um2::Point<D> p0 = seg[0];
  Float l = 0;
  for (Int i = 0; i < num_segs; ++i) {
    Float const r1 = r + dr;
    um2::Point<D> const p1 = seg(r1);
    l += p0.distanceTo(p1);
    p0 = p1;
    r = r1;
  }
  ASSERT_NEAR(l, seg.length(), eps);
}

template <Int D>
HOSTDEV
TEST_CASE(length)
{
  testLength(makeSeg1<D>());
  testLength(makeSeg2<D>());
  testLength(makeSeg3<D>());
  testLength(makeSeg4<D>());
  testLength(makeSeg5<D>());
  testLength(makeSeg6<D>());
  testLength(makeSeg7<D>());
  testLength(makeSeg8<D>());
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticSegment<D> seg1 = makeSeg1<D>();
  um2::AxisAlignedBox<D> const bb1 = seg1.boundingBox();
  um2::AxisAlignedBox<D> const bb_ref(seg1[0], seg1[1]);
  ASSERT(bb1.isApprox(bb_ref));

  um2::QuadraticSegment<D> seg2 = makeSeg2<D>();
  um2::AxisAlignedBox<D> const bb2 = seg2.boundingBox();
  um2::AxisAlignedBox<D> bb_ref2(seg2[0], seg2[1]);
  bb_ref2 += seg2[2];
  ASSERT(bb2.isApprox(bb_ref2));

  um2::QuadraticSegment<D> const seg8 = makeSeg8<D>();
  um2::AxisAlignedBox<D> const bb8 = seg8.boundingBox();
  um2::Point<D> const p0 = um2::Vec<D, Float>::zero();
  um2::Point<D> p1 = um2::Vec<D, Float>::zero();
  p1[0] = castIfNot<Float>(4.083334);
  p1[1] = castIfNot<Float>(3);
  um2::AxisAlignedBox<D> const bb_ref8(p0, p1);
  ASSERT(bb8.isApprox(bb_ref8));
}

//==============================================================================
// isLeft
//==============================================================================

HOSTDEV
TEST_CASE(isLeft)
{
  um2::Vector<um2::Point2> const test_points = {
      um2::Point2(castIfNot<Float>(1), castIfNot<Float>(3)),      // always left
      um2::Point2(castIfNot<Float>(1), castIfNot<Float>(-3)),     // always right
      um2::Point2(castIfNot<Float>(-1), ahalf),              // always left
      um2::Point2(castIfNot<Float>(-1), castIfNot<Float>(-0.5)),  // always right
      um2::Point2(castIfNot<Float>(3), ahalf),               // always left
      um2::Point2(castIfNot<Float>(3), castIfNot<Float>(-0.5)),   // always right
      um2::Point2(castIfNot<Float>(0.1), castIfNot<Float>(0.9)),  // always left
      um2::Point2(castIfNot<Float>(0.1), castIfNot<Float>(-0.9)), // always right
      um2::Point2(castIfNot<Float>(1.9), castIfNot<Float>(0.9)),  // always left
      um2::Point2(castIfNot<Float>(1.9), castIfNot<Float>(-0.9)), // always right
      um2::Point2(castIfNot<Float>(1.1), ahalf),
      um2::Point2(castIfNot<Float>(2), ahalf),
      um2::Point2(castIfNot<Float>(2.1), castIfNot<Float>(0.01)),
      um2::Point2(castIfNot<Float>(2.1), ahalf),
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
      um2::Point2(castIfNot<Float>(1), castIfNot<Float>(3)),      // always left
      um2::Point2(castIfNot<Float>(1), castIfNot<Float>(-3)),     // always right
      um2::Point2(castIfNot<Float>(-1), ahalf),              // always left
      um2::Point2(castIfNot<Float>(-1), castIfNot<Float>(-0.5)),  // always right
      um2::Point2(castIfNot<Float>(3), ahalf),               // always left
      um2::Point2(castIfNot<Float>(3), castIfNot<Float>(-0.5)),   // always right
      um2::Point2(castIfNot<Float>(0.1), castIfNot<Float>(0.9)),  // always left
      um2::Point2(castIfNot<Float>(0.1), castIfNot<Float>(-0.9)), // always right
      um2::Point2(castIfNot<Float>(1.9), castIfNot<Float>(0.9)),  // always left
      um2::Point2(castIfNot<Float>(1.9), castIfNot<Float>(-0.9)), // always right
      um2::Point2(castIfNot<Float>(1.1), ahalf),
      um2::Point2(castIfNot<Float>(2), ahalf),
      um2::Point2(castIfNot<Float>(2.1), castIfNot<Float>(0.01)),
      um2::Point2(castIfNot<Float>(2.1), ahalf),
  };
  auto const big_eps = castIfNot<Float>(1e-2);
  um2::Vector<um2::QuadraticSegment2> const segments = {
      makeSeg1<2>(), makeSeg2<2>(), makeSeg3<2>(), makeSeg4<2>(),
      makeSeg5<2>(), makeSeg6<2>(), makeSeg7<2>(), makeSeg8<2>(),
  };

  for (auto const & q : segments) {
    for (um2::Point2 const & p : test_points) {
      Float const r0 = q.pointClosestTo(p);
      Float const d0 = p.distanceTo(q(r0));
      Float const r1 = r0 - big_eps;
      Float const d1 = p.distanceTo(q(r1));
      Float const r2 = r0 + big_eps;
      Float const d2 = p.distanceTo(q(r2));
      std::cerr << "r0 = " << r0 << ", r1 = " << r1 << ", r2 = " << r2 << std::endl;
      std::cerr << "d0 = " << d0 << ", d1 = " << d1 << ", d2 = " << d2 << std::endl;
      if (0 <= r1 && r1 <= 1) {
        ASSERT(d0 < d1);
      }
      if (0 <= r2 && r2 <= 1) {
        std::cerr << "d0 = " << d0 << ", d2 = " << d2 << std::endl;
        ASSERT(d0 < d2);
      }
    }
  }
}

//==============================================================================----------
// enclosedArea
//==============================================================================----------

HOSTDEV
void
testEnclosedArea(um2::QuadraticSegment2 const & q)
{
  // Shoot vertical rays from the bottom of the bounding box to
  // perform Riemann sum to compute the area.
  Int constexpr nrays = 1000;

  auto aabb = q.boundingBox();
  auto const dx =  aabb.width() / static_cast<Float>(nrays);
  um2::Vec2F const dir(0, 1); 
  um2::Vec2F origin = aabb.minima();
  origin[0] -= dx / 2;
  Float area = 0;
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2 const ray(origin, dir);
    auto intersections = q.intersect(ray);
    // Sort intersections
    if (intersections[0] > intersections[1]) {
      um2::swap(intersections[0], intersections[1]);
    }
    // Two valid intersections
    if (intersections[1] < um2::inf_distance / 10) {
      auto const p0 = ray(intersections[0]);
      auto const p1 = ray(intersections[1]);
      auto const d = p0.distanceTo(p1); 
      area += d * dx;
    // 1 valid intersection
    } else if (intersections[0] < um2::inf_distance / 10) {
      auto const p0 = ray.origin();
      auto const p1 = ray(intersections[0]);
      auto const d = p0.distanceTo(p1);
      area += d * dx;
    }
  }
  // Compare with the area computed by the function
  // The computed area should be negative since we assume that
  // curving left is positive, and all of these segments curve right.
  // This is because a segment which is curving left will produce a convex
  // polygon when CCW oriented, and a segment which is curving right will
  // produce a concave polygon when CCW oriented.
  auto const area_computed = -enclosedArea(q);
  auto const err = um2::abs(area_computed - area) / area;
  // Less than 1% error
  ASSERT(err < 1e-2);
}

HOSTDEV
TEST_CASE(enclosedArea)
{
  // testEnclosedArea only works if the segment is bounded below
  // by the x-axis
  um2::QuadraticSegment2 const seg1 = makeSeg1<2>();
  Float const area = enclosedArea(seg1);
  auto const area_ref = castIfNot<Float>(0);
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(area, area_ref, eps);

  testEnclosedArea(makeSeg2<2>());
  testEnclosedArea(makeSeg4<2>());
  testEnclosedArea(makeSeg6<2>());
  testEnclosedArea(makeSeg8<2>());

  // Check that the negative versions of the segments produce the same area
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(-enclosedArea(makeSeg2<2>()), enclosedArea(makeSeg3<2>()), eps);
  ASSERT_NEAR(-enclosedArea(makeSeg4<2>()), enclosedArea(makeSeg5<2>()), eps);
  ASSERT_NEAR(-enclosedArea(makeSeg6<2>()), enclosedArea(makeSeg7<2>()), eps);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================----------
// enclosedCentroid
//==============================================================================----------

HOSTDEV
void
testEnclosedCentroid(um2::QuadraticSegment2 const & q)
{
  // Shoot vertical rays from the bottom of the bounding box to
  // perform Riemann sum to compute the area.
  // Use geometric decomposition to compute the centroid.
  Int constexpr nrays = 1000;

  auto aabb = q.boundingBox();
  auto const dx =  aabb.width() / static_cast<Float>(nrays);
  um2::Vec2F const dir(0, 1); 
  um2::Vec2F origin = aabb.minima();
  origin[0] -= dx / 2;
  Float area = 0;
  um2::Vec2F centroid = um2::Vec2F::zero();
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2 const ray(origin, dir);
    auto intersections = q.intersect(ray);
    // Sort intersections
    if (intersections[0] > intersections[1]) {
      um2::swap(intersections[0], intersections[1]);
    }
    // Two valid intersections
    if (intersections[1] < um2::inf_distance / 10) {
      auto const p0 = ray(intersections[0]);
      auto const p1 = ray(intersections[1]);
      auto const d = p0.distanceTo(p1); 
      auto const p_center = um2::midpoint(p0, p1);
      auto const area_segment = d * dx;
      area += area_segment; 
      centroid += area_segment * p_center;
    // 1 valid intersection
    } else if (intersections[0] < um2::inf_distance / 10) {
      auto const p0 = ray.origin();
      auto const p1 = ray(intersections[0]);
      auto const d = p0.distanceTo(p1);
      auto const p_center = um2::midpoint(p0, p1);
      auto const area_segment = d * dx;
      area += area_segment;
      centroid += area_segment * p_center;
    }
  }
  centroid /= area;
  auto const centroid_computed = enclosedCentroid(q);
  auto const err_x = um2::abs(centroid_computed[0] - centroid[0]) / centroid[0];
  auto const err_y = um2::abs(centroid_computed[1] - centroid[1]) / centroid[1];
  // Less than 1% error
  ASSERT(err_x < 1e-2);
  ASSERT(err_y < 1e-2);
}

HOSTDEV
TEST_CASE(enclosedCentroid)
{
  // testEnclosedCentroid only works if the segment is bounded below
  // by the x-axis
  um2::QuadraticSegment2 const seg1 = makeSeg1<2>();
  auto const centroid = enclosedCentroid(seg1);
  auto const centroid_ref = um2::Point2(1, 0);
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT(centroid.isApprox(centroid_ref));

  testEnclosedCentroid(makeSeg2<2>());
  testEnclosedCentroid(makeSeg4<2>());
  testEnclosedCentroid(makeSeg6<2>());
  testEnclosedCentroid(makeSeg8<2>());

  // Check that we get the right answer for a segment that is translated and rotated
  um2::QuadraticSegment2 seg6 = makeSeg6<2>();
  auto centroid6 = enclosedCentroid(seg6);
  // Rotated 240 degrees + translated by (1, 1)
  um2::Mat2x2F const rot = um2::rotationMatrix(4 * um2::pi<Float> / 3);
  um2::Vec2F const trans = um2::Vec2F(1, 1);
  seg6[0] = rot * seg6[0] + trans;
  seg6[1] = rot * seg6[1] + trans;
  seg6[2] = rot * seg6[2] + trans;
  auto centroid6_rot = enclosedCentroid(seg6);
  centroid6 = rot * centroid6 + trans;
  ASSERT(centroid6.isApprox(centroid6_rot));
}

HOSTDEV
void
testEdgeForIntersections(um2::QuadraticSegment2 const & q)
{
  // Parameters
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto constexpr eps_pt = 1e-2;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      auto intersections = q.intersect(ray);
      for (Int j = 0; j < 2; ++j) {
        Float const r = intersections[j];
        if (r < um2::inf_distance / 10) {
          um2::Point2 const p = ray(r);
          Float const d = q.distanceTo(p);
          if (d > eps_pt) {
            std::cerr << "j = " << j << std::endl;
            std::cerr << "ray origin = (" << ray.origin()[0] << ", " << ray.origin()[1] << ")" << std::endl;
            std::cerr << "ray direction = (" << ray.direction()[0] << ", " << ray.direction()[1] << ")" << std::endl;
            std::cerr << "d = " << d << std::endl;
            std::cerr << "r = " << r << std::endl;
            std::cerr << "p = (" << p[0] << ", " << p[1] << ")" << std::endl;
          }
//          ASSERT(d < eps_pt);
        }
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  std::cout << "Seg2" << std::endl;
  testEdgeForIntersections(makeSeg2<2>());
  std::cout << "Seg3" << std::endl;
  testEdgeForIntersections(makeSeg3<2>());
  std::cout << "Seg4" << std::endl;
  testEdgeForIntersections(makeSeg4<2>());
  std::cout << "Seg5" << std::endl;
  testEdgeForIntersections(makeSeg5<2>());
  std::cout << "Seg6" << std::endl;
  testEdgeForIntersections(makeSeg6<2>());
  std::cout << "Seg7" << std::endl;
  testEdgeForIntersections(makeSeg7<2>());
  std::cout << "Seg8" << std::endl;
  testEdgeForIntersections(makeSeg8<2>());
}

template <Int D>
HOSTDEV
void
testPolyCoeffs(um2::QuadraticSegment<D> const & q)
{
  auto const coeffs = q.getPolyCoeffs();
  auto const c = coeffs[0];
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  // Check against interpolation
  Int constexpr num_points = 100;
  for (Int i = 0; i < num_points; ++i) {
    Float const r = static_cast<Float>(i) / static_cast<Float>(num_points - 1);
    auto const p_ref = q(r);
    auto const p = c + r * b + (r * r) * a;
    ASSERT(p.isApprox(p_ref));
  }
}

template <Int D>
HOSTDEV
TEST_CASE(getPolyCoeffs)
{
  testPolyCoeffs(makeSeg1<D>());
  testPolyCoeffs(makeSeg2<D>());
  testPolyCoeffs(makeSeg3<D>());
  testPolyCoeffs(makeSeg4<D>());
  testPolyCoeffs(makeSeg5<D>());
  testPolyCoeffs(makeSeg6<D>());
  testPolyCoeffs(makeSeg7<D>());
  testPolyCoeffs(makeSeg8<D>());
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);

template <Int D>
MAKE_CUDA_KERNEL(jacobian, D);

template <Int D>
MAKE_CUDA_KERNEL(length, D);

template <Int D>
MAKE_CUDA_KERNEL(boundingBox, D);

MAKE_CUDA_KERNEL(isLeft);

MAKE_CUDA_KERNEL(enclosedArea);

MAKE_CUDA_KERNEL(enclosedCentroid);

MAKE_CUDA_KERNEL(pointClosestTo);
#endif

template <Int D>
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(boundingBox, D);
  TEST_HOSTDEV(length, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isLeft);
    TEST_HOSTDEV(pointClosestTo);
    TEST_HOSTDEV(enclosedArea);
    TEST_HOSTDEV(enclosedCentroid);
    TEST_HOSTDEV(intersect);
  }
  TEST_HOSTDEV(getPolyCoeffs, D);
}

auto
main() -> int
{
  RUN_SUITE(QuadraticSegment<2>);
  RUN_SUITE(QuadraticSegment<3>);
  return 0;
}
