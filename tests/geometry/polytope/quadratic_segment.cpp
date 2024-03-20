#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/modular_rays.hpp>

#include "../../test_macros.hpp"

#include <random>
#include <iostream>

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
void
testIsLeft(um2::QuadraticSegment2 const & q)
{
  Int constexpr num_points = 10000;

  auto const aabb_tight = q.boundingBox();
  auto aabb = q.boundingBox();
  aabb.scale(2);
  auto const width = aabb.width();
  auto const height = aabb.height();
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Float> dis(0, 1);

  auto const coeffs = q.getPolyCoeffs();
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  for (Int i = 0; i < num_points; ++i) {
    Float const x = aabb.xMin() + dis(gen) * width;
    Float const y = aabb.yMin() + dis(gen) * height;
    um2::Point2 const p(x, y);
    // Check if the point is to the left or right of the segment
    bool const is_left = q.isLeft(p);
    // If the point is in the tight bounding box, then to confirm, 
    // get the point on the segment that is closest to p.
    // Then, check if the cross product of the tangent vector at p_closest
    // and the vector from p_closest to p is positive or negative.
    // If it is positive, then p is to the left of the segment.
    //
    // If the point is not in the tight bounding box, then we simply check
    // (p1 - p0) x (p - p0) > 0
    if (aabb_tight.contains(p)) {
      Float const r = q.pointClosestTo(p);
      um2::Point2 const p_closest = q(r);
      // Q(r) = C + rB + r^2A -> Q'(r) = B + 2rA
      um2::Vec2F const vtan = b + (2 * r) * a;
      bool const is_left_ref = vtan.cross(p - p_closest) >= 0; 
      if (is_left != is_left_ref) {
        std::cerr << "q[0] = (" << q[0][0] << ", " << q[0][1] << ")\n";
        std::cerr << "q[1] = (" << q[1][0] << ", " << q[1][1] << ")\n";
        std::cerr << "q[2] = (" << q[2][0] << ", " << q[2][1] << ")\n";
        std::cerr << "aabb_tight = (" << aabb_tight.xMin() << ", " << aabb_tight.yMin() << ", " << aabb_tight.xMax() << ", " << aabb_tight.yMax() << ")\n";
        std::cerr << "p = (" << p[0] << ", " << p[1] << ")\n";
        std::cerr << "r = " << r << "\n";
        std::cerr << "p_closest = (" << p_closest[0] << ", " << p_closest[1] << ")\n";
        std::cerr << "vtan = (" << vtan[0] << ", " << vtan[1] << ")\n";
        std::cerr << "p - p_closest = (" << p[0] - p_closest[0] << ", " << p[1] - p_closest[1] << ")\n";
        std::cerr << "is_left = " << is_left << "\n";
        std::cerr << "is_left_ref = " << is_left_ref << "\n";
        std::cerr << "vtan.cross(p - p_closest) = " << vtan.cross(p - p_closest) << "\n";
      }
      ASSERT(is_left == is_left_ref);
    } else {
      bool const is_left_ref = (q[1] - q[0]).cross(p - q[0]) > 0;
      ASSERT(is_left == is_left_ref);
    }
  }
}

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

  testIsLeft(q2);
  testIsLeft(q3);
  testIsLeft(q4);
  testIsLeft(q5);
  testIsLeft(q6);
  testIsLeft(q7);
  testIsLeft(q8);
}

//==============================================================================
// pointClosestTo
//==============================================================================

HOSTDEV
void
testPoint(um2::QuadraticSegment2 const & q, um2::Point2 const p)
{
  auto constexpr dr = castIfNot<Float>(1e-4);
  Float const r = q.pointClosestTo(p);
  um2::Point2 const p_closest = q(r);
  Float const d_closest = p.distanceTo(p_closest);

  if (0 <= (r + dr) && (r + dr) <= 1) {
    um2::Point2 const p_plus = q(r + dr);
    Float const d_plus = p.distanceTo(p_plus);
    ASSERT(d_closest <= d_plus);
  }
  if (0 <= (r - dr) && (r - dr) <= 1) {
    um2::Point2 const p_minus = q(r - dr);
    Float const d_minus = p.distanceTo(p_minus);
    ASSERT(d_closest <= d_minus);
  }
}

HOSTDEV
void
testPCT(um2::QuadraticSegment2 const & q)
{
  // For a number of points on the bounding box of the segment,
  // find the point on the segment that is closest to the point.
  // Perturb the parametric coordinate of the point in the + and - directions.
  // If either of the perturbed points is closer to the box point than the
  // original point, then the test fails.
  Int constexpr points_per_side = 100;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const dx =  aabb.width() / static_cast<Float>(points_per_side - 1);
  // Bottom side
  for (Int i = 0; i < points_per_side; ++i) {
    Float const x = aabb.xMin() + i * dx;
    Float const y = aabb.yMin();
    um2::Point2 const p(x, y);
    testPoint(q, p);
  }
  // Top side
  for (Int i = 0; i < points_per_side; ++i) {
    Float const x = aabb.xMin() + i * dx;
    Float const y = aabb.yMax();
    um2::Point2 const p(x, y);
    testPoint(q, p);
  }
  auto const dy =  aabb.height() / static_cast<Float>(points_per_side - 1);
  // Left side
  for (Int i = 0; i < points_per_side; ++i) {
    Float const x = aabb.xMin();
    Float const y = aabb.yMin() + i * dy;
    um2::Point2 const p(x, y);
    testPoint(q, p);
  }
  // Right side
  for (Int i = 0; i < points_per_side; ++i) {
    Float const x = aabb.xMax();
    Float const y = aabb.yMin() + i * dy;
    um2::Point2 const p(x, y);
    testPoint(q, p);
  }
}

HOSTDEV
TEST_CASE(pointClosestTo)
{
  testPCT(makeSeg1<2>());
  testPCT(makeSeg2<2>());
  testPCT(makeSeg3<2>());
  testPCT(makeSeg4<2>());
  testPCT(makeSeg5<2>());
  testPCT(makeSeg6<2>());
  testPCT(makeSeg7<2>());
  testPCT(makeSeg8<2>());
}

//==============================================================================
// enclosedArea
//==============================================================================

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
    // Sort intersections biggest to smallest
    if (intersections[0] < intersections[1]) {
      um2::swap(intersections[0], intersections[1]);
    }
    // Two valid intersections
    if (0 < intersections[1]) {
      auto const p0 = ray(intersections[0]);
      auto const p1 = ray(intersections[1]);
      auto const d = p0.distanceTo(p1);
      area += d * dx;
    // 1 valid intersection
    } else if (0 < intersections[0]) {
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

//==============================================================================
// enclosedCentroid
//==============================================================================

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
    // Sort intersections biggest to smallest
    if (intersections[0] < intersections[1]) {
      um2::swap(intersections[0], intersections[1]);
    }
    // Two valid intersections
    if (0 < intersections[1]) {
      auto const p0 = ray(intersections[0]);
      auto const p1 = ray(intersections[1]);
      auto const d = p0.distanceTo(p1);
      auto const p_center = um2::midpoint(p0, p1);
      auto const area_segment = d * dx;
      area += area_segment;
      centroid += area_segment * p_center;
    // 1 valid intersection
    } else if (0 < intersections[0]) {
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
  um2::Mat2x2F const rot = um2::makeRotationMatrix(4 * um2::pi<Float> / 3);
  um2::Vec2F const trans = um2::Vec2F(1, 1);
  seg6[0] = rot * seg6[0] + trans;
  seg6[1] = rot * seg6[1] + trans;
  seg6[2] = rot * seg6[2] + trans;
  auto centroid6_rot = enclosedCentroid(seg6);
  centroid6 = rot * centroid6 + trans;
  ASSERT(centroid6.isApprox(centroid6_rot));
}

//==============================================================================
// intersect
//==============================================================================

HOSTDEV
void
testEdgeForIntersections(um2::QuadraticSegment2 const & q)
{
  // Parameters
  // Tested up to 128, 10000 with
  // um2::abs(p) < 3e-6 || um2::abs(q_over_p) > 1500
  // on the cubic equation single root case
  Int constexpr num_angles = 32; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 1000;

  auto constexpr eps_pt = 1e-4;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.width() > aabb.height() ? aabb.width() : aabb.height();
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(num_angles);
//  uint64_t tested_rays = 0;
  // For each angle
  for (Int ia = 0; ia < num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
//      ++tested_rays; 
      auto const ray = params.getRay(i);
      auto intersections = q.intersect(ray);
      for (Int j = 0; j < 2; ++j) {
        Float const r = intersections[j];
        if (0 <= r) {
          um2::Point2 const p = ray(r);
          Float const s = q.pointClosestTo(p);
          um2::Point2 const q_closest = q(s);
          Float const d = q_closest.distanceTo(p);
          ASSERT(d < eps_pt);
        }
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  testEdgeForIntersections(makeSeg2<2>());
  testEdgeForIntersections(makeSeg3<2>());
  testEdgeForIntersections(makeSeg4<2>());
  testEdgeForIntersections(makeSeg5<2>());
  testEdgeForIntersections(makeSeg6<2>());
  testEdgeForIntersections(makeSeg7<2>());
  testEdgeForIntersections(makeSeg8<2>());
}

//==============================================================================
// testPolyCoeffs
//==============================================================================

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
    auto const p = c + r * (b + r * a);
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
