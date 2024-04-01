#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/modular_rays.hpp>

#include "../../test_macros.hpp"

#include <random>

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
Float constexpr eps = um2::eps_distance;
Float constexpr ahalf = castIfNot<Float>(1) / castIfNot<Float>(2);
Int constexpr num_angles = 16; // Number of angles for rotation

template <Int D>
HOSTDEV constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D>
{
  um2::QuadraticSegment<D> q;
  q[0] = 0;
  q[1] = 0;
  q[2] = 0;
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

template <Int D>
HOSTDEV constexpr auto
getSeg(Int i) -> um2::QuadraticSegment<D>
{
  switch (i) {
    case 1: return makeSeg1<D>();
    case 2: return makeSeg2<D>();
    case 3: return makeSeg3<D>();
    case 4: return makeSeg4<D>();
    case 5: return makeSeg5<D>();
    case 6: return makeSeg6<D>();
    case 7: return makeSeg7<D>();
    case 8: return makeSeg8<D>();
    default: return makeBaseSeg<D>();
  }
}

HOSTDEV void
rotate(um2::QuadraticSegment2 & q, Float const angle)
{
  um2::Mat2x2F const rot = um2::makeRotationMatrix(angle);
  q[0] = rot * q[0];
  q[1] = rot * q[1];
  q[2] = rot * q[2];
}

void
perturb(um2::QuadraticSegment2 & q)
{
  auto constexpr delta = castIfNot<Float>(0.25);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  static std::mt19937 gen(seed);
  static std::uniform_real_distribution<Float> dis(-delta, delta);
  q[0][0] += dis(gen);
  q[0][1] += dis(gen);
  q[1][0] += dis(gen);
  q[1][1] += dis(gen);
  q[2][0] += dis(gen);
  q[2][1] += dis(gen);
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
  // Straight segment
  um2::QuadraticSegment<D> const seg = makeSeg1<D>();
  um2::Vec<D, Float> j0 = seg.jacobian(0);
  um2::Vec<D, Float> j12 = seg.jacobian(static_cast<Float>(0.5));
  um2::Vec<D, Float> j1 = seg.jacobian(1);
  um2::Vec<D, Float> j_ref;
  j_ref = 0; 
  j_ref[0] = static_cast<Float>(2);

  ASSERT(j0.isApprox(j_ref));
  ASSERT(j12.isApprox(j_ref));
  ASSERT(j1.isApprox(j_ref));

  um2::QuadraticSegment<D> const seg2 = makeSeg2<D>();
  j0 = seg2.jacobian(0);
  j12 = seg2.jacobian(static_cast<Float>(0.5));
  j1 = seg2.jacobian(1);
  ASSERT_NEAR(j0[0], 2, eps); 
  ASSERT(j0[1] > 0);
  ASSERT_NEAR(j12[0], 2, eps);
  ASSERT_NEAR(j12[1], 0, eps);
  ASSERT_NEAR(j1[0],  2, eps); 
  ASSERT(j1[1] < 0);
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
  // Allow for 0.1% error
  ASSERT(um2::abs(l - seg.length()) / l < castIfNot<Float>(1e-3));
}

template <Int D>
HOSTDEV
TEST_CASE(length)
{
  if constexpr (D == 2) {
    Int const num_perturb = 10;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment<D> seg = getSeg<D>(iseg);
          Float const angle = static_cast<Float>(iang * 2) * um2::pi<Float> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          testLength(seg);
        }
      }
    }
  } else {
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      um2::QuadraticSegment<D> const seg = getSeg<D>(iseg);
      testLength(seg);
    }
  }
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

  if constexpr (D == 2) {
    Int const num_perturb = 10;
    Int const num_interp = 100;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment2 seg = getSeg<2>(iseg);
          Float const angle = static_cast<Float>(iang * 2) * um2::pi<Float> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          um2::AxisAlignedBox2 bb = seg.boundingBox();
          // Account or floating point error by scaling the box by 1%
          bb.scale(castIfNot<Float>(1.01));
          for (Int i = 0; i <= num_interp; ++i) {
            Float const r = castIfNot<Float>(i) / castIfNot<Float>(num_interp);
            um2::Point2 const p = seg(r);
            ASSERT(bb.contains(p));
          }
        }
      }
    }
  }
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Int D>
HOSTDEV
void
testPoint(um2::QuadraticSegment<D> const & q, um2::Point<D> const p)
{
  auto constexpr dr = castIfNot<Float>(1e-3);
  Float const r = q.pointClosestTo(p);
  um2::Point<D> const p_closest = q(r);
  Float const d_closest = p.distanceTo(p_closest);

  if (0 <= (r + dr) && (r + dr) <= 1) {
    um2::Point<D> const p_plus = q(r + dr);
    Float const d_plus = p.distanceTo(p_plus);
    ASSERT(d_closest <= d_plus);
  }
  if (0 <= (r - dr) && (r - dr) <= 1) {
    um2::Point<D> const p_minus = q(r - dr);
    Float const d_minus = p.distanceTo(p_minus);
    ASSERT(d_closest <= d_minus);
  }
}

template <Int D>
void
testPCT(um2::QuadraticSegment<D> const & q)
{
  // For a number of points in the bounding box of the segment,
  // find the point on the segment that is closest to the point.
  // Perturb the parametric coordinate of the point in the + and - directions.
  // If either of the perturbed points is closer to the box point than the
  // original point, then the test fails.
  Int constexpr num_points = 100;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(0x08FA9A20);
  if constexpr (D == 2) {
    std::uniform_real_distribution<Float> dis_x(aabb.minima(0), aabb.maxima(0));
    std::uniform_real_distribution<Float> dis_y(aabb.minima(1), aabb.maxima(1));
    for (Int i = 0; i < num_points; ++i) {
      um2::Point2 const p(dis_x(gen), dis_y(gen));
      testPoint(q, p);
    }
  } else {
    std::uniform_real_distribution<Float> dis_x(aabb.minima(0), aabb.maxima(0));
    std::uniform_real_distribution<Float> dis_y(aabb.minima(1), aabb.maxima(1));
    std::uniform_real_distribution<Float> dis_z(aabb.minima(2), aabb.maxima(2));
    for (Int i = 0; i < num_points; ++i) {
      um2::Point3 const p(dis_x(gen), dis_y(gen), dis_z(gen));
      testPoint(q, p);
    }
  }
}

template <Int D>
HOSTDEV
TEST_CASE(pointClosestTo)
{
  if constexpr (D == 2) {
    Int const num_perturb = 10;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment2 seg = getSeg<2>(iseg);
          Float const angle = static_cast<Float>(iang * 2) * um2::pi<Float> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          testPCT(seg);
        }
      }
    }
  } else {
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      um2::QuadraticSegment<D> const seg = getSeg<D>(iseg);
      testPCT(seg);
    }
  }
}

//==============================================================================
// isLeft
//==============================================================================

HOSTDEV
void
testIsLeft(um2::QuadraticSegment2 const & q)
{
  Int constexpr num_points = 1000;

  auto const aabb_tight = q.boundingBox();
  um2::Point2 const bcp = um2::getBezierControlPoint(q);
  um2::Triangle2 tri(q[0], q[1], bcp);
  if (!tri.isCCW()) {
    tri.flip();
  }
  ASSERT(tri.isCCW());

  auto aabb = q.boundingBox();
  aabb.scale(2);
  auto const width = aabb.extents(0);
  auto const height = aabb.extents(1);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Float> dis(0, 1);

  auto const coeffs = q.getPolyCoeffs();
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  for (Int i = 0; i < num_points; ++i) {
    Float const x = aabb.minima(0) + dis(gen) * width;
    Float const y = aabb.minima(1) + dis(gen) * height;
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
    if (aabb_tight.contains(p) && tri.contains(p)) {
      Float const r = q.pointClosestTo(p);
      um2::Point2 const p_closest = q(r);
      // Q(r) = C + rB + r^2A -> Q'(r) = B + 2rA
      um2::Vec2F const vtan = b + (2 * r) * a;
      bool const is_left_ref = vtan.cross(p - p_closest) >= 0;
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

  Int const num_perturb = 10;
  for (Int iseg = 2; iseg <= 8; ++iseg) {
    for (Int iang = 0; iang < num_angles; ++iang) {
      for (Int ipert = 0; ipert < num_perturb; ++ipert) {
        um2::QuadraticSegment2 seg = getSeg<2>(iseg);
        Float const angle = static_cast<Float>(iang * 2) * um2::pi<Float> / num_angles;
        rotate(seg, angle);
        perturb(seg);
        testIsLeft(seg);
      }
    }
  }
}

//==============================================================================
// intersect
//==============================================================================

HOSTDEV
void
testEdgeForIntersections(um2::QuadraticSegment2 const & q)
{
  // Parameters
  Int constexpr intersect_num_angles = 16; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 200;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<Float>(1.1));
  auto const longest_edge = aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<Float>(rays_per_longest_edge);
  Float const pi_deg = um2::pi_2<Float> / static_cast<Float>(intersect_num_angles);
  // For each angle
  for (Int ia = 0; ia < intersect_num_angles; ++ia) {
    Float const angle = pi_deg * static_cast<Float>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      Float buf[2];
      auto const hits = q.intersect(ray, buf);
      for (Int ihit = 0; ihit < hits; ++ihit) {
        Float const r = buf[ihit]; 
        um2::Point2 const p = ray(r);
        Float const s = q.pointClosestTo(p);
        um2::Point2 const q_closest = q(s);
        Float const d = q_closest.distanceTo(p);
        // Add additional tolerance for 32-bit floating point
#if UM2_ENABLE_FLOAT64
        ASSERT(d < eps);
#else
        ASSERT(d < 70 * eps);
#endif
      }
    }
  }
}

HOSTDEV
TEST_CASE(intersect)
{
  Int const num_perturb = 10;
  for (Int iseg = 1; iseg <= 8; ++iseg) {
    for (Int iang = 0; iang < num_angles; ++iang) {
      for (Int ipert = 0; ipert < num_perturb; ++ipert) {
        um2::QuadraticSegment2 seg = getSeg<2>(iseg);
        Float const angle = static_cast<Float>(iang * 2) * um2::pi<Float> / num_angles;
        rotate(seg, angle);
        perturb(seg);
        testEdgeForIntersections(seg);
      }
    }
  }
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
  auto const dx =  aabb.extents(0) / static_cast<Float>(nrays);
  um2::Vec2F const dir(0, 1);
  um2::Vec2F origin = aabb.minima();
  origin[0] -= dx / 2;
  Float area = 0;
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2 const ray(origin, dir);
    Float buf[2];
    auto const hits = q.intersect(ray, buf);
    if (hits == 0) {
      continue;
    }
    if (hits == 1) {
      auto const p0 = ray.origin();
      auto const p1 = ray(buf[0]);
      auto const d = p0.distanceTo(p1);
      area += d * dx;
    }
    if (hits == 2) {
      if (buf[0] < buf[1]) {
        um2::swap(buf[0], buf[1]);
      }
      auto const p0 = ray(buf[0]);
      auto const p1 = ray(buf[1]);
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
  // Less than 0.1% error
  ASSERT(err < castIfNot<Float>(1e-3));
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
  auto const dx =  aabb.extents(0) / static_cast<Float>(nrays);
  um2::Vec2F const dir(0, 1);
  um2::Vec2F origin = aabb.minima();
  origin[0] -= dx / 2;
  Float area = 0;
  um2::Vec2F centroid = um2::Vec2F::zero();
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2 const ray(origin, dir);
    Float buf[2];
    auto const hits = q.intersect(ray, buf);
    if (hits == 0) {
      continue;
    }
    if (hits == 1) {
      auto const p0 = ray.origin();
      auto const p1 = ray(buf[0]);
      auto const d = p0.distanceTo(p1);
      auto const p_center = um2::midpoint(p0, p1);
      auto const area_segment = d * dx;
      area += area_segment;
      centroid += area_segment * p_center;
    }
    if (hits == 2) {
      if (buf[0] < buf[1]) {
        um2::swap(buf[0], buf[1]);
      }
      auto const p0 = ray(buf[0]);
      auto const p1 = ray(buf[1]);
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
  // Less than 0.1% error
  ASSERT(err_x < castIfNot<Float>(1e-3));
  ASSERT(err_y < castIfNot<Float>(1e-3));
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

template <Int D>
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, D);
  TEST_HOSTDEV(jacobian, D);
  TEST_HOSTDEV(getPolyCoeffs, D);
  TEST_HOSTDEV(length, D);
  TEST_HOSTDEV(boundingBox, D);
  TEST_HOSTDEV(pointClosestTo, D);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isLeft);
    TEST_HOSTDEV(intersect);
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
