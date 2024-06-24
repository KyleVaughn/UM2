#include <um2/config.hpp>
#include <um2/geometry/modular_rays.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/geometry/quadratic_segment.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/mat.hpp>
#include <um2/math/vec.hpp>
#include <um2/stdlib/vector.hpp>

#include "../../test_macros.hpp"

#include <random>
#include <cstdint>

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
template <class T>
T constexpr eps = um2::epsDistance<T>();

template <class T>
T constexpr ahalf = castIfNot<T>(1) / castIfNot<T>(2);

Int constexpr num_angles = 16; // Number of angles for rotation

template <Int D, class T>
HOSTDEV constexpr auto
makeBaseSeg() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q;
  q[0] = 0;
  q[1] = 0;
  q[2] = 0;
  q[1][0] = castIfNot<T>(2);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg1() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg2() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(1);
  q[2][1] = castIfNot<T>(1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg3() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(1);
  q[2][1] = castIfNot<T>(-1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg4() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(2);
  q[2][1] = castIfNot<T>(1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg5() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(2);
  q[2][1] = castIfNot<T>(-1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg6() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(0);
  q[2][1] = castIfNot<T>(1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg7() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[2][0] = castIfNot<T>(0);
  q[2][1] = castIfNot<T>(-1);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
makeSeg8() -> um2::QuadraticSegment<D, T>
{
  um2::QuadraticSegment<D, T> q = makeBaseSeg<D, T>();
  q[1][0] = castIfNot<T>(2);
  q[2][0] = castIfNot<T>(4);
  q[2][1] = castIfNot<T>(3);
  return q;
}

template <Int D, class T>
HOSTDEV constexpr auto
getSeg(Int i) -> um2::QuadraticSegment<D, T>
{
  switch (i) {
  case 1:
    return makeSeg1<D, T>();
  case 2:
    return makeSeg2<D, T>();
  case 3:
    return makeSeg3<D, T>();
  case 4:
    return makeSeg4<D, T>();
  case 5:
    return makeSeg5<D, T>();
  case 6:
    return makeSeg6<D, T>();
  case 7:
    return makeSeg7<D, T>();
  case 8:
    return makeSeg8<D, T>();
  default:
    return makeBaseSeg<D, T>();
  }
}

template <class T>
HOSTDEV void
rotate(um2::QuadraticSegment2<T> & q, T const angle)
{
  um2::Mat2x2<T> const rot = um2::makeRotationMatrix(angle);
  q[0] = rot * q[0];
  q[1] = rot * q[1];
  q[2] = rot * q[2];
}

template <class T>
void
perturb(um2::QuadraticSegment2<T> & q)
{
  auto constexpr delta = castIfNot<T>(0.25);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  static std::mt19937 gen(seed);
  static std::uniform_real_distribution<T> dis(-delta, delta);
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

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::QuadraticSegment<D, T> const seg = makeSeg2<D, T>();
  for (Int i = 0; i < 5; ++i) {
    T const r = castIfNot<T>(i) / castIfNot<T>(4);
    um2::Point<D, T> const p = seg(r);
    um2::Point<D, T> p_ref = um2::Vec<D, T>::zero();
    p_ref[0] = 2 * r;
    p_ref[1] = 4 * r * (1 - r);
    ASSERT(p.isApprox(p_ref));
  }
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(jacobian)
{
  // Straight segment
  um2::QuadraticSegment<D, T> const seg = makeSeg1<D, T>();
  um2::Vec<D, T> j0 = seg.jacobian(0);
  um2::Vec<D, T> j12 = seg.jacobian(castIfNot<T>(0.5));
  um2::Vec<D, T> j1 = seg.jacobian(1);
  um2::Vec<D, T> j_ref;
  j_ref = 0;
  j_ref[0] = castIfNot<T>(2);

  ASSERT(j0.isApprox(j_ref));
  ASSERT(j12.isApprox(j_ref));
  ASSERT(j1.isApprox(j_ref));

  um2::QuadraticSegment<D, T> const seg2 = makeSeg2<D, T>();
  j0 = seg2.jacobian(0);
  j12 = seg2.jacobian(castIfNot<T>(0.5));
  j1 = seg2.jacobian(1);
  ASSERT_NEAR(j0[0], 2, eps<T>);
  ASSERT(j0[1] > 0);
  ASSERT_NEAR(j12[0], 2, eps<T>);
  ASSERT_NEAR(j12[1], 0, eps<T>);
  ASSERT_NEAR(j1[0], 2, eps<T>);
  ASSERT(j1[1] < 0);
}

//==============================================================================
// testPolyCoeffs
//==============================================================================

template <Int D, class T>
HOSTDEV void
testPolyCoeffs(um2::QuadraticSegment<D, T> const & q)
{
  auto const coeffs = q.getPolyCoeffs();
  auto const c = coeffs[0];
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  // Check against interpolation
  Int constexpr num_points = 100;
  for (Int i = 0; i < num_points; ++i) {
    T const r = static_cast<T>(i) / static_cast<T>(num_points - 1);
    auto const p_ref = q(r);
    auto const p = c + r * (b + r * a);
    ASSERT(p.isApprox(p_ref));
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(getPolyCoeffs)
{
  testPolyCoeffs(makeSeg1<D, T>());
  testPolyCoeffs(makeSeg2<D, T>());
  testPolyCoeffs(makeSeg3<D, T>());
  testPolyCoeffs(makeSeg4<D, T>());
  testPolyCoeffs(makeSeg5<D, T>());
  testPolyCoeffs(makeSeg6<D, T>());
  testPolyCoeffs(makeSeg7<D, T>());
  testPolyCoeffs(makeSeg8<D, T>());
}

//==============================================================================
// length
//==============================================================================

template <Int D, class T>
HOSTDEV void
testLength(um2::QuadraticSegment<D, T> const & seg)
{
  // Note: this samples evenly in parametric space, not in physical space.
  Int constexpr num_segs = 10000;
  T const dr = castIfNot<T>(1) / castIfNot<T>(num_segs);
  T r = 0;
  um2::Point<D, T> p0 = seg[0];
  T l = 0;
  for (Int i = 0; i < num_segs; ++i) {
    T const r1 = r + dr;
    um2::Point<D, T> const p1 = seg(r1);
    l += p0.distanceTo(p1);
    p0 = p1;
    r = r1;
  }
  // Allow for 0.1% error
  ASSERT(um2::abs(l - seg.length()) / l < castIfNot<T>(1e-3));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(length)
{
  if constexpr (D == 2) {
    Int const num_perturb = 10;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment<D, T> seg = getSeg<D, T>(iseg);
          T const angle = static_cast<T>(iang * 2) * um2::pi<T> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          testLength(seg);
        }
      }
    }
  } else {
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      um2::QuadraticSegment<D, T> const seg = getSeg<D, T>(iseg);
      testLength(seg);
    }
  }
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(boundingBox)
{
  um2::QuadraticSegment<D, T> seg1 = makeSeg1<D, T>();
  um2::AxisAlignedBox<D, T> const bb1 = seg1.boundingBox();
  um2::AxisAlignedBox<D, T> const bb_ref(seg1[0], seg1[1]);
  ASSERT(bb1.isApprox(bb_ref));

  um2::QuadraticSegment<D, T> seg2 = makeSeg2<D, T>();
  um2::AxisAlignedBox<D, T> const bb2 = seg2.boundingBox();
  um2::AxisAlignedBox<D, T> bb_ref2(seg2[0], seg2[1]);
  bb_ref2 += seg2[2];
  ASSERT(bb2.isApprox(bb_ref2));

  um2::QuadraticSegment<D, T> const seg8 = makeSeg8<D, T>();
  um2::AxisAlignedBox<D, T> const bb8 = seg8.boundingBox();
  um2::Point<D, T> const p0 = um2::Vec<D, T>::zero();
  um2::Point<D, T> p1 = um2::Vec<D, T>::zero();
  p1[0] = castIfNot<T>(4.0833333);
  p1[1] = castIfNot<T>(3);
  um2::AxisAlignedBox<D, T> const bb_ref8(p0, p1);
  ASSERT(bb8.isApprox(bb_ref8));

  if constexpr (D == 2) {
    Int const num_perturb = 10;
    Int const num_interp = 100;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment2<T> seg = getSeg<2, T>(iseg);
          T const angle = static_cast<T>(iang * 2) * um2::pi<T> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          um2::AxisAlignedBox2<T> bb = seg.boundingBox();
          // Account or floating point error by scaling the box by 1%
          bb.scale(castIfNot<T>(1.01));
          for (Int i = 0; i <= num_interp; ++i) {
            T const r = castIfNot<T>(i) / castIfNot<T>(num_interp);
            um2::Point2<T> const p = seg(r);
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

template <Int D, class T>
HOSTDEV void
testPoint(um2::QuadraticSegment<D, T> const & q, um2::Point<D, T> const p)
{
  auto constexpr dr = castIfNot<T>(1e-3);
  T const r = q.pointClosestTo(p);
  um2::Point<D, T> const p_closest = q(r);
  T const d_closest = p.distanceTo(p_closest);

  if (0 <= (r + dr) && (r + dr) <= 1) {
    um2::Point<D, T> const p_plus = q(r + dr);
    T const d_plus = p.distanceTo(p_plus);
    ASSERT(d_closest <= d_plus);
  }
  if (0 <= (r - dr) && (r - dr) <= 1) {
    um2::Point<D, T> const p_minus = q(r - dr);
    T const d_minus = p.distanceTo(p_minus);
    ASSERT(d_closest <= d_minus);
  }
}

template <Int D, class T>
void
testPCT(um2::QuadraticSegment<D, T> const & q)
{
  // For a number of points in the bounding box of the segment,
  // find the point on the segment that is closest to the point.
  // Perturb the parametric coordinate of the point in the + and - directions.
  // If either of the perturbed points is closer to the box point than the
  // original point, then the test fails.
  Int constexpr num_points = 100;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<T>(1.1));
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937 gen(0x08FA9A20);
  if constexpr (D == 2) {
    std::uniform_real_distribution<T> dis_x(aabb.minima(0), aabb.maxima(0));
    std::uniform_real_distribution<T> dis_y(aabb.minima(1), aabb.maxima(1));
    for (Int i = 0; i < num_points; ++i) {
      um2::Point2<T> const p(dis_x(gen), dis_y(gen));
      testPoint(q, p);
    }
  } else {
    std::uniform_real_distribution<T> dis_x(aabb.minima(0), aabb.maxima(0));
    std::uniform_real_distribution<T> dis_y(aabb.minima(1), aabb.maxima(1));
    std::uniform_real_distribution<T> dis_z(aabb.minima(2), aabb.maxima(2));
    for (Int i = 0; i < num_points; ++i) {
      um2::Point3<T> const p(dis_x(gen), dis_y(gen), dis_z(gen));
      testPoint(q, p);
    }
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(pointClosestTo)
{
  if constexpr (D == 2) {
    Int const num_perturb = 10;
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      for (Int iang = 0; iang < num_angles; ++iang) {
        for (Int ipert = 0; ipert < num_perturb; ++ipert) {
          um2::QuadraticSegment2<T> seg = getSeg<2, T>(iseg);
          T const angle = static_cast<T>(iang * 2) * um2::pi<T> / num_angles;
          rotate(seg, angle);
          perturb(seg);
          testPCT(seg);
        }
      }
    }
  } else {
    for (Int iseg = 1; iseg <= 8; ++iseg) {
      um2::QuadraticSegment<D, T> const seg = getSeg<D, T>(iseg);
      testPCT(seg);
    }
  }
}

//==============================================================================
// isLeft
//==============================================================================

template <class T>
HOSTDEV
void
testIsLeft(um2::QuadraticSegment2<T> const & q)
{
  Int constexpr num_points = 1000;

  auto const aabb_tight = q.boundingBox();
  um2::Point2<T> const bcp = um2::getBezierControlPoint(q);
  um2::Triangle2<T> tri(q[0], q[1], bcp);
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
  std::uniform_real_distribution<T> dis(0, 1);

  auto const coeffs = q.getPolyCoeffs();
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  for (Int i = 0; i < num_points; ++i) {
    T const x = aabb.minima(0) + dis(gen) * width;
    T const y = aabb.minima(1) + dis(gen) * height;
    um2::Point2<T> const p(x, y);
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
      T const r = q.pointClosestTo(p);
      um2::Point2<T> const p_closest = q(r);
      // Q(r) = C + rB + r^2A -> Q'(r) = B + 2rA
      um2::Vec2<T> const vtan = b + (2 * r) * a;
      bool const is_left_ref = vtan.cross(p - p_closest) >= 0;
      ASSERT(is_left == is_left_ref);
    } else {
      bool const is_left_ref = (q[1] - q[0]).cross(p - q[0]) > 0;
      ASSERT(is_left == is_left_ref);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(isLeft)
{
  um2::Vector<um2::Point2<T>> const test_points = {
      um2::Point2<T>(castIfNot<T>(1), castIfNot<T>(3)),      // always left
      um2::Point2<T>(castIfNot<T>(1), castIfNot<T>(-3)),     // always right
      um2::Point2<T>(castIfNot<T>(-1), ahalf<T>),                   // always left
      um2::Point2<T>(castIfNot<T>(-1), castIfNot<T>(-0.5)),  // always right
      um2::Point2<T>(castIfNot<T>(3), ahalf<T>),                    // always left
      um2::Point2<T>(castIfNot<T>(3), castIfNot<T>(-0.5)),   // always right
      um2::Point2<T>(castIfNot<T>(0.1), castIfNot<T>(0.9)),  // always left
      um2::Point2<T>(castIfNot<T>(0.1), castIfNot<T>(-0.9)), // always right
      um2::Point2<T>(castIfNot<T>(1.9), castIfNot<T>(0.9)),  // always left
      um2::Point2<T>(castIfNot<T>(1.9), castIfNot<T>(-0.9)), // always right
      um2::Point2<T>(castIfNot<T>(1.1), ahalf<T>),
      um2::Point2<T>(castIfNot<T>(2), ahalf<T>),
      um2::Point2<T>(castIfNot<T>(2.1), castIfNot<T>(0.01)),
      um2::Point2<T>(castIfNot<T>(2.1), ahalf<T>),
  };

  // A straight line
  um2::QuadraticSegment2<T> const q1 = makeSeg1<2, T>();
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
  um2::QuadraticSegment2<T> const q2 = makeSeg2<2, T>();
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
  um2::QuadraticSegment2<T> const q3 = makeSeg3<2, T>();
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
  um2::QuadraticSegment2<T> const q4 = makeSeg4<2, T>();
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
  um2::QuadraticSegment2<T> const q5 = makeSeg5<2, T>();
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
  um2::QuadraticSegment2<T> const q6 = makeSeg6<2, T>();
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
  um2::QuadraticSegment2<T> const q7 = makeSeg7<2, T>();
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
  um2::QuadraticSegment2<T> const q8 = makeSeg8<2, T>();
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
        um2::QuadraticSegment2<T> seg = getSeg<2, T>(iseg);
        T const angle = static_cast<T>(iang * 2) * um2::pi<T> / num_angles;
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

template <class T>
HOSTDEV
void
testEdgeForIntersections(um2::QuadraticSegment2<T> const & q)
{
  // Parameters
  Int constexpr intersect_num_angles = 16; // Angles γ ∈ (0, π).
  Int constexpr rays_per_longest_edge = 200;

  auto aabb = q.boundingBox();
  aabb.scale(castIfNot<T>(1.1));
  auto const longest_edge =
      aabb.extents(0) > aabb.extents(1) ? aabb.extents(0) : aabb.extents(1);
  auto const spacing = longest_edge / static_cast<T>(rays_per_longest_edge);
  T const pi_deg = um2::pi_2<T> / static_cast<T>(intersect_num_angles);
  // For each angle
  for (Int ia = 0; ia < intersect_num_angles; ++ia) {
    T const angle = pi_deg * static_cast<T>(2 * ia + 1);
    // Compute modular ray parameters
    um2::ModularRayParams const params(angle, spacing, aabb);
    Int const num_rays = params.getTotalNumRays();
    // For each ray
    for (Int i = 0; i < num_rays; ++i) {
      auto const ray = params.getRay(i);
      T buf[2];
      auto const hits = q.intersect(ray, buf);
      for (Int ihit = 0; ihit < hits; ++ihit) {
        T const r = buf[ihit];
        um2::Point2<T> const p = ray(r);
        T const s = q.pointClosestTo(p);
        um2::Point2<T> const q_closest = q(s);
        T const d = q_closest.distanceTo(p);
        ASSERT(d < eps<T>);
      }
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(intersect)
{
  Int const num_perturb = 10;
  for (Int iseg = 1; iseg <= 8; ++iseg) {
    for (Int iang = 0; iang < num_angles; ++iang) {
      for (Int ipert = 0; ipert < num_perturb; ++ipert) {
        um2::QuadraticSegment2<T> seg = getSeg<2, T>(iseg);
        T const angle = static_cast<T>(iang * 2) * um2::pi<T> / num_angles;
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

template <class T>
HOSTDEV
void
testEnclosedArea(um2::QuadraticSegment2<T> const & q)
{
  // Shoot vertical rays from the bottom of the bounding box to
  // perform Riemann sum to compute the area.
  Int constexpr nrays = 1000;

  auto aabb = q.boundingBox();
  auto const dx = aabb.extents(0) / static_cast<T>(nrays);
  um2::Vec2<T> const dir(0, 1);
  um2::Vec2<T> origin = aabb.minima();
  origin[0] -= dx / 2;
  T area = 0;
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2<T> const ray(origin, dir);
    T buf[2];
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
  ASSERT(err < castIfNot<T>(1e-3));
}

template <class T>
HOSTDEV
TEST_CASE(enclosedArea)
{
  // testEnclosedArea only works if the segment is bounded below
  // by the x-axis
  um2::QuadraticSegment2<T> const seg1 = makeSeg1<2, T>();
  T const area = enclosedArea(seg1);
  auto const area_ref = castIfNot<T>(0);
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(area, area_ref, eps<T>);

  testEnclosedArea(makeSeg2<2, T>());
  testEnclosedArea(makeSeg4<2, T>());
  testEnclosedArea(makeSeg6<2, T>());
  testEnclosedArea(makeSeg8<2, T>());

  // Check that the negative versions of the segments produce the same area
  // NOLINTBEGIN(cert-dcl03-c,misc-static-assert)
  ASSERT_NEAR(-enclosedArea(makeSeg2<2, T>()), enclosedArea(makeSeg3<2, T>()), eps<T>);
  ASSERT_NEAR(-enclosedArea(makeSeg4<2, T>()), enclosedArea(makeSeg5<2, T>()), eps<T>);
  ASSERT_NEAR(-enclosedArea(makeSeg6<2, T>()), enclosedArea(makeSeg7<2, T>()), eps<T>);
  // NOLINTEND(cert-dcl03-c,misc-static-assert)
}

//==============================================================================
// enclosedCentroid
//==============================================================================

template <class T>
HOSTDEV
void
testEnclosedCentroid(um2::QuadraticSegment2<T> const & q)
{
  // Shoot vertical rays from the bottom of the bounding box to
  // perform Riemann sum to compute the area.
  // Use geometric decomposition to compute the centroid.
  Int constexpr nrays = 1000;

  auto aabb = q.boundingBox();
  auto const dx = aabb.extents(0) / static_cast<T>(nrays);
  um2::Vec2<T> const dir(0, 1);
  um2::Vec2<T> origin = aabb.minima();
  origin[0] -= dx / 2;
  T area = 0;
  um2::Vec2<T> centroid = um2::Vec2<T>::zero();
  for (Int i = 0; i < nrays; ++i) {
    origin[0] += dx;
    um2::Ray2<T> const ray(origin, dir);
    T buf[2];
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
  ASSERT(err_x < castIfNot<T>(1e-3));
  ASSERT(err_y < castIfNot<T>(1e-3));
}

template <class T>
HOSTDEV
TEST_CASE(enclosedCentroid)
{
  // testEnclosedCentroid only works if the segment is bounded below
  // by the x-axis
  um2::QuadraticSegment2<T> const seg1 = makeSeg1<2, T>();
  auto const centroid = enclosedCentroid(seg1);
  auto const centroid_ref = um2::Point2<T>(1, 0);
  // NOLINTNEXTLINE(cert-dcl03-c,misc-static-assert)
  ASSERT(centroid.isApprox(centroid_ref));

  testEnclosedCentroid(makeSeg2<2, T>());
  testEnclosedCentroid(makeSeg4<2, T>());
  testEnclosedCentroid(makeSeg6<2, T>());
  testEnclosedCentroid(makeSeg8<2, T>());

  // Check that we get the right answer for a segment that is translated and rotated
  um2::QuadraticSegment2<T> seg6 = makeSeg6<2, T>();
  auto centroid6 = enclosedCentroid(seg6);
  // Rotated 240 degrees + translated by (1, 1)
  um2::Mat2x2<T> const rot = um2::makeRotationMatrix(4 * um2::pi<T> / 3);
  um2::Vec2<T> const trans = um2::Vec2<T>(1, 1);
  seg6[0] = rot * seg6[0] + trans;
  seg6[1] = rot * seg6[1] + trans;
  seg6[2] = rot * seg6[2] + trans;
  auto centroid6_rot = enclosedCentroid(seg6);
  centroid6 = rot * centroid6 + trans;
  ASSERT(centroid6.isApprox(centroid6_rot));
}

template <class T>
HOSTDEV
TEST_CASE(intersect_quadratic_segment)
{
  //============================================================================
  // No intersection
  //============================================================================

  // parallel straight lines
  //---------------------------------------------------------------------------
  um2::QuadraticSegment2<T> seg1;
  seg1[0] = um2::Point2<T>(0, 0);
  seg1[1] = um2::Point2<T>(2, 0);
  seg1[2] = um2::Point2<T>(1, 0);

  um2::Point2<T> offset;
  offset[0] = 0;
  offset[1] = castIfNot<T>(0.5);

  um2::QuadraticSegment2<T> seg2;
  seg2[0] = seg1[0] + offset;
  seg2[1] = seg1[1] + offset;
  seg2[2] = seg1[2] + offset;

  um2::Point2<T> buf[8];
  Int hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 0);
  ASSERT(!seg1.intersects(seg2));
  ASSERT(!seg2.intersects(seg1));

  // parallel quadratic segments with overlapping bounding boxes
  //---------------------------------------------------------------------------
  seg1[2] = um2::Point2<T>(1, 1);
  seg2[2] = seg1[2] + offset;
  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 0);
  ASSERT(!seg1.intersects(seg2));
  ASSERT(!seg2.intersects(seg1));

  // parallel quadratic segments that are VERY close to each other
  //---------------------------------------------------------------------------
  offset[0] = 0;
  offset[1] = 10 * um2::epsDistance<T>();
  seg2[0] = seg1[0] + offset;
  seg2[1] = seg1[1] + offset;
  seg2[2] = seg1[2] + offset;
  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 0);
  ASSERT(!seg1.intersects(seg2));
  ASSERT(!seg2.intersects(seg1));

  // quadratic segments that intersect outside of r,s ∈ [0, 1]
  //---------------------------------------------------------------------------
  seg1[0] = um2::Point2<T>(0, 0);
  seg1[1] = um2::Point2<T>(2, 0);
  seg1[2] = um2::Point2<T>(1, 1);

  seg2[0] = um2::Point2<T>(1, 0);
  seg2[1] = um2::Point2<T>(3, 0);
  seg2[2] = um2::Point2<T>(2, -1);

  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 0);
  ASSERT(!seg1.intersects(seg2));
  ASSERT(!seg2.intersects(seg1));

  //============================================================================
  // One intersection
  //============================================================================

  // intersection at midpoint of two straight lines
  //---------------------------------------------------------------------------
  seg1[0] = um2::Point2<T>(0, 0);
  seg1[1] = um2::Point2<T>(2, 0);
  seg1[2] = um2::Point2<T>(1, 0);

  seg2[0] = um2::Point2<T>(1, 1);
  seg2[1] = um2::Point2<T>(1, -1);
  seg2[2] = um2::Point2<T>(1, 0);

  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 1);
  ASSERT(buf[0].isApprox(um2::Point2<T>(1, 0)));
  ASSERT(seg1.intersects(seg2));
  ASSERT(seg2.intersects(seg1));

  // intersection at endpoint of two curved segments
  //---------------------------------------------------------------------------
  seg2[0] = um2::Point2<T>(2, 0);
  seg2[1] = um2::Point2<T>(1, 3);
  seg2[2] = um2::Point2<T>(2, 1);

  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 1);
  ASSERT(buf[0].isApprox(um2::Point2<T>(2, 0)));
  ASSERT(seg1.intersects(seg2));
  ASSERT(seg2.intersects(seg1));

  //============================================================================
  // Two intersections
  //============================================================================

  seg1[0] = um2::Point2<T>(0, 0);
  seg1[1] = um2::Point2<T>(3, 0);
  seg1[2] = um2::Point2<T>(2, 1);

  seg2[0] = um2::Point2<T>(3, 0);
  seg2[1] = um2::Point2<T>(0, 1);
  seg2[2] = um2::Point2<T>(2, -1);

  hits = um2::intersect(seg1, seg2, buf);
  ASSERT(hits == 2);
  um2::Point2<T> const p0(castIfNot<T>(0.48), castIfNot<T>(0.36));
  um2::Point2<T> const p1(3, 0);
  ASSERT(buf[0].isApprox(p0));
  ASSERT(buf[1].isApprox(p1));
  ASSERT(seg1.intersects(seg2));
  ASSERT(seg2.intersects(seg1));

  hits = seg1.intersect(seg2, buf);
  ASSERT(hits == 2);
  ASSERT(buf[0].isApprox(p0));
  ASSERT(buf[1].isApprox(p1));
  ASSERT(seg1.intersects(seg2));
  ASSERT(seg2.intersects(seg1));

  hits = seg2.intersect(seg1, buf);
  ASSERT(hits == 2);
  ASSERT(buf[0].isApprox(p1));
  ASSERT(buf[1].isApprox(p0));
  ASSERT(seg1.intersects(seg2));
  ASSERT(seg2.intersects(seg1));
}

template <Int D, class T>
TEST_SUITE(QuadraticSegment)
{
  TEST_HOSTDEV(interpolate, D, T);
  TEST_HOSTDEV(jacobian, D, T);
  TEST_HOSTDEV(getPolyCoeffs, D, T);
  TEST_HOSTDEV(length, D, T);
  TEST_HOSTDEV(boundingBox, D, T);
  TEST_HOSTDEV(pointClosestTo, D, T);
  if constexpr (D == 2) {
    TEST_HOSTDEV(isLeft, T);
    TEST_HOSTDEV(intersect, T);
    TEST_HOSTDEV(enclosedArea, T);
    TEST_HOSTDEV(enclosedCentroid, T);
    TEST_HOSTDEV(intersect_quadratic_segment, T);
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
