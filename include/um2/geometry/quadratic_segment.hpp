#pragma once

#include <um2/geometry/polytope.hpp>
#include <um2/geometry/line_segment.hpp>
#include <um2/geometry/triangle.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/math/cubic_equation.hpp>

//==============================================================================
// QUADRATIC SEGMENT
//==============================================================================
//
// For quadratic segments, the parametric equation is
//  Q(r) = P₁ + rB + r²A,
// where
//  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
//  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
// and
// V₁₃ = q[3] - q[1]
// V₂₃ = q[3] - q[2]
// NOTE: The equations above use 1-based indexing.

namespace um2
{

template <Int D>
class Polytope<1, 2, 3, D>
{
  static_assert(0 < D && D <= 3, "Only 1D, 2D, and 3D segments are supported.");

public:
  using Vertex = Point<D>;

private:
  Vertex _v[3];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> Vertex &;

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> Vertex const &;

  // Returns a pointer to the vertex array
  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vertex const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
  requires(sizeof...(Pts) == 3 && (std::same_as<Vertex, Pts> && ...))
      // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
      HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  HOSTDEV constexpr explicit Polytope(Vec<3, Vertex> const & v) noexcept;

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the segment.
  // r in [0, 1] are valid values.
  // Float(r) -> (x, y, z)
  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Vertex;

  // dF/dr (r) -> (dx/dr, dy/dr, dz/dr)
  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r) const noexcept -> Vec<D, Float>;

  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // The rotation matrix is returned.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat2x2F
  requires(D == 2);

  // If a point is to the left of the segment, with the segment oriented from
  // r = 0 to r = 1.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Vertex const & p) const noexcept -> bool requires(D == 2);

  // Arc length of the segment
  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // Return the point on the curve that is closest to the given point.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Vertex const & p) const noexcept -> Float;

  // Return the squared distance between the given point and the closest point
  // on the curve.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vertex const & p) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vertex const & p) const noexcept -> Float;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  intersect(Ray2 ray) const noexcept -> Vec2F
  requires(D == 2);

  PURE HOSTDEV [[nodiscard]] constexpr auto
  curvesLeft() const noexcept -> bool requires(D == 2);

  // Q(r) = C + rB + r²A
  // returns {C, B, A}
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPolyCoeffs() const noexcept -> Vec<3, Vertex>;

}; // QuadraticSegment

//==============================================================================
// Free functions
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D> const & q) noexcept -> bool;

// The area enclosed by the segment and the straight line from the p0 to p1.
PURE HOSTDEV constexpr auto
enclosedArea(QuadraticSegment2 const & q) noexcept -> Float;

// The centroid of the area enclosed by the segment and the straight line from
// the p0 to p1.
PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2 const & q) noexcept -> Vec2F;

//==============================================================================
// Constructors
//==============================================================================

template <Int D>
HOSTDEV constexpr QuadraticSegment<D>::Polytope(Vec<3, Vertex> const & v) noexcept
{
  _v[0] = v[0];
  _v[1] = v[1];
  _v[2] = v[2];
}

//==============================================================================
// Accessors
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::operator[](Int i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::operator[](Int i) const noexcept -> Vertex const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < 3);
  return _v[i];
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::vertices() const noexcept -> Vertex const *
{
  return _v;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Int D>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::operator()(R const r) const noexcept -> Vertex
{
  // Q(r) =
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  auto const rr = static_cast<Float>(r);
  Float const two_rr_1 = 2 * rr - 1;
  Float const rr_1 = rr - 1;

  Float const w0 = two_rr_1 * rr_1;
  Float const w1 = two_rr_1 * rr;
  Float const w2 = -4 * rr * rr_1;
  return w0 * _v[0] + w1 * _v[1] + w2 * _v[2];
}

//==============================================================================
// jacobian
//==============================================================================

template <Int D>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::jacobian(R const r) const noexcept -> Point<D>
{
  // Q'(r) = B + 2rA
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  Float const w0 = 4 * static_cast<Float>(r) - 3;
  Float const w1 = 4 * static_cast<Float>(r) - 1;
  return w0 * (_v[0] - _v[2]) + w1 * (_v[1] - _v[2]);
}

//==============================================================================
// getRotation
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::getRotation() const noexcept -> Mat2x2F
requires(D == 2) {
  return LineSegment2(_v[0], _v[1]).getRotation();
}

//==============================================================================
// getBezierControlPoint
//==============================================================================
// Get the control point for the quadratic Bezier curve that interpolates the
// given quadratic segment.

template <Int D>
PURE HOSTDEV static constexpr auto
getBezierControlPoint(QuadraticSegment<D> const & q) noexcept -> Point<D>
{
  // p0 == v[0]
  // p2 == v[1]
  // p1 == 2 * v[2] - (v[0] + v[1]) / 2, hence we only need to compute p1
  return 2 * q[2] - (q[0] + q[1]) / 2;
}

//==============================================================================
// pointIsLeft
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::isLeft(Vertex const & p) const noexcept -> bool requires(D == 2)
{
  // The quadratic segment has an equivalent quadratic Bezier curve.
  // The triangle formed by the control points of the quadratic Bezier curve
  // bound the curve. If p is not in this triangle we can ignore the curvature of the
  // segment and return simply based upon areCCW(v[0], v[1], p).
  Vec2F const bcp = getBezierControlPoint(*this);
  Triangle2 tri(_v[0], _v[1], bcp);
  // precondition for tri.contains(p) is that tri.isCCW()
  if (!tri.isCCW()) {
    tri.flip();
  }
  ASSERT(tri.isCCW());
  if (!tri.contains(p)) {
    return areCCW(_v[0], _v[1], p);
  }

  // The point is in the bounding triangle, so we must perform further analysis.
  // To make the math easier, we transform the coordinate system so that v[0] is at the
  // origin and v[1] is on the x-axis. In particular, we want v[1][0] > 0.

  QuadraticSegment2 q(Point2(0, 0), _v[1] - _v[0], _v[2] - _v[0]);
  Mat2x2F const rot = q.getRotation();
  q[1] = rot * q[1];
  q[2] = rot * q[2];
  ASSERT(um2::abs(q[1][1]) < eps_distance);
  ASSERT(q[1][0] > 0);
  auto p_r = rot * (p - _v[0]);

  // Orient the segment so that it curves to the right. (positive y-coordinat for v[2])
  bool const curves_right = q[2][1] >= 0;
  if (!curves_right) {
    // Flip the y-coordinates to be greater than or equal to zero
    q[2][1] = -q[2][1];
    p_r[1] = -p_r[1];
  }
  ASSERT(q[1][0] > 0);
  ASSERT(q[2][1] >= 0);

  // Now, since we may have flipped the y-coordinates, we will always need to
  // account for this when we return the result.
  // | isLeft | curves_right | final result |
  // |--------|--------------|--------------|
  // | true   | true         | true         |
  // | false  | true         | false        |
  // | true   | false        | false        |
  // | false  | false        | true         |
  // Or, simply: curves_right ? is_left : !is_left

  // The point must be above the origin if it is in the bounding triangle.
  ASSERT(p_r[1] >= 0);
  //  // If the point is below the origin, then the point is right of the segment
  //  if (p_r[1] < 0) {
  //    return !curves_right;
  //  }

  // If q[2][1] < eps_distance, then the segment is straight and we can exit early.
  // Note q[2][1] is positive.
  if (q[2][1] <= eps_distance) {
    // LineSegment2(q[0], q[1]).isLeft(p_r) is equivalent to areCCW(q[0], q[1], p_r)
    // areCCW(q[0], q[1], p_r) is equivalent to 0 <= (q[1] - q[0]).cross(p_r - q[0])
    // Since q[0] == (0, 0), we can simplify the cross product to q[1].cross(p_r).
    // q[1].cross(p_r) is equivalent to q[1][0] * p_r[1] - q[1][1] * p_r[0]
    // Since q[1][1] == 0, we can simplify the cross product to q[1][0] * p_r[1]
    // Since q[1][0] > 0, we can simplify further to p_r[1] >= 0
    bool const is_left = p_r[1] >= 0;
    return curves_right ? is_left : !is_left;
  }

  // Otherwise, there is non-trivial curvature in the segment.
  // Find rx such that Q(rx)[0] = p_r[0].
  // Use the y-coordinate/coordinates of the point/points to check if the point 
  // is to the left of the segment.
  // We must also check that rx is in [0, 1] to ensure that the point is valid.
  //
  // Q(r) = C + rB + r²A
  // In this case: 
  // C = q[0] = (0, 0)
  // B = -q[1] + 4q[2]
  // A = 2q[1] - 4q[2] = -B + q[1]
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // 0 = c + br + ar²
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A_x == 0, then we have a well-behaved quadratic segment, and there is one root.
  // This is the expected case.

  Float const bx = -q[1][0] + 4 * q[2][0];
  Float const by = 4 * q[2][1]; // q[1][1] == 0, q[2][1] > 0 implies by > 0
  Float const ax = -bx + q[1][0]; 
  Float const ay = -by; // q[1][1] == 0, by > 0 implies ay < 0
  ASSERT(by > 0);

  if (um2::abs(ax) < 4 * eps_distance) {
    // Hence we wish to solve 0 = -P_x + rB_x for r.
    ASSERT(um2::abs(bx) > eps_distance);
    Float const r = p_r[0] / bx; // bx != 0, otherwise the segment would be degenerate.
    // We know the point is in the bounding triangle, so we expect r to be in [0, 1]
    ASSERT(0 <= r);
    ASSERT(r <= 1);
    Float const qy = r * (by + r * ay);
    bool const is_left = p_r[1] >= qy;
    return curves_right ? is_left : !is_left;
  }

  // Two roots
  Float const disc = bx * bx + 4 * ax * p_r[0];
  ASSERT_ASSUME(disc >= 0);
  Float const r1 = (-bx + um2::sqrt(disc)) / (2 * ax);
  Float const r2 = (-bx - um2::sqrt(disc)) / (2 * ax);
  Float const qy1 = r1 * (by + r1 * ay);
  Float const qy2 = r2 * (by + r2 * ay);
  Float const qymin = um2::min(qy1, qy2);
  Float const qymax = um2::max(qy1, qy2);
  bool const contained_in_curve = qymin <= p_r[1] && p_r[1] <= qymax;
  bool const is_left = !contained_in_curve;
  return curves_right ? is_left : !is_left;
}

//==============================================================================
// length
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::length() const noexcept -> Float
{
  // The arc length integral may be reduced to an integral over the square root of a
  // quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
  //              1             1
  // arc length = ∫ ‖Q′(r)‖dr = ∫ √(ar² + br + c) dr
  //              0             0
  //
  // Q(r) = x0 + r * x1 + r² * x2,
  // Q′(r) = x1 + 2r * x2,

  if (isStraight(*this)) {
    return _v[0].distanceTo(_v[1]);
  }

  auto const coeffs = getPolyCoeffs();
  auto const x1 = coeffs[1];
  auto const x2 = coeffs[2];

  // ‖Q′(r)‖ =  √(4(x2 ⋅x2)r² + 4(x2 ⋅x1)r + x1 ⋅x1) = √(ar² + br + c)
  // where
  // a = 4(x2 ⋅ x2)
  // b = 4(x2 ⋅ x1)
  // c = x1 ⋅ x1

  ASSERT(squaredNorm(x2) > eps_distance2);
  Float const a = 4 * x2.squaredNorm();
  Float const b = 4 * x2.dot(x1);
  Float const c = x1.squaredNorm();

  // According to Wolfram Alpha, the integral of √(ax² + bx + c) is
  // ((b + 2 a x) sqrt(c + x (b + a x)))/(4 a)
  // - ((b^2 - 4 a c) log(b + 2 a x + 2 sqrt(a) sqrt(c + x (b + a x))))/(8 a^(3/2))
  // + constant
  //
  // Simplifying the integral:
  // ((b^2 - 4 a c) (log(2 sqrt(a) sqrt(c) + b) - log(2 sqrt(a) sqrt(a + b + c) + 2 a + b))
  //  + sqrt(a) (2 (2 a + b) sqrt(a + b + c) - 2 b sqrt(c)))/(8 a^(3/2))

  // sa = sqrt(a)
  // sb = sqrt(b)
  // sc = sqrt(c)
  // sabc = sqrt(a + b + c)
  // disc = b^2 - 4 a c
  // a2b = 2 a + b

  // (disc * (log(2 * sa * sc + b)
  //  - log(2 * sa * sabc + a2b))
  //  + sa * (2 * (a2b * sabc - b * sc))
  //  ) / (8 * sa * sa * sa)
  //
  Float const sa = um2::sqrt(a);
  Float const sc = um2::sqrt(c);
  Float const sabc = um2::sqrt(a + b + c);
  Float const disc = b * b - 4 * a * c;
  Float const a2b = 2 * a + b;
  Float const num = disc * (um2::log(2 * sa * sc + b) - um2::log(2 * sa * sabc + a2b)) + sa * (2 * (a2b * sabc - b * sc));
  Float const den = 8 * sa * sa * sa;
  Float const result  = num / den;
  ASSERT(0 <= result);
  ASSERT(result <= inf_distance);
  return result;
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  // Floatind the extrema by finding dx_i/dr = 0
  // Q(r) = C + rB + r²A,
  // Q′(r) = B + 2rA,
  // 0 = B + 2rA ==> (r_i,...) = -B / (2A)
  // x_i = Q(r_i) = C + (r_i)B + (r_i)^2 A = C - B² / (4A)
  // Compare the extrema with the segment's endpoints to find the AABox
  auto const coeffs = getPolyCoeffs();
  auto const b = coeffs[1];
  auto const a = coeffs[2];
  Point<D> minima = um2::min(_v[0], _v[1]);
  Point<D> maxima = um2::max(_v[0], _v[1]);
  for (Int i = 0; i < D; ++i) {
    // If a[i] is small, then x_i varies linearly with r, so the extrema are the
    // end points and we can skip the computation.
    if (um2::abs(a[i]) < 4 * eps_distance) {
      continue;
    }
    Float const half_b = b[i] / 2;
    Float const r = - half_b / a[i];
    // if r is not in [0, 1], then the extrema are not on the segment, hence
    // the segment's endpoints are the extrema.
    if (0 < r && r < 1) {
      // x_i = Q(r_i) = C - B² / (4A) = C - r_i(B/2)
      Float const x = _v[0][i] + r * half_b;
      minima[i] = um2::min(minima[i], x);
      maxima[i] = um2::max(maxima[i], x);
    }
  }
  return AxisAlignedBox<D>{minima, maxima};
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::pointClosestTo(Vertex const & p) const noexcept -> Float
{

  if (isStraight(*this)) {
    return LineSegment<D>(_v[0], _v[1]).pointClosestTo(p);
  }

  // The interpolation function of the quadratic segment is
  // Q(r) = C + rB + r²A,
  //
  // We wish to find r which minimizes ‖Q(r) - P‖.
  // This r also minimizes ‖Q(r) - P‖².
  // ‖Q(r) - P‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
  // Let W = C - P
  // a₄ = A ⋅ A
  // a₃ = 2(A ⋅ B)
  // a₂ = 2(A ⋅ W) + (B ⋅ B)
  // a₁ = 2(B ⋅ W)
  // a₀ = W ⋅ W
  //
  // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
  // a = 2(A ⋅ A)
  // b = 3(A ⋅ B)
  // c = (B ⋅ B) + 2(A ⋅W)
  // d = (B ⋅ W)
  // Note we factored out a 2
  auto const coeffs = getPolyCoeffs();
  auto const vc = coeffs[0];
  auto const vb = coeffs[1];
  auto const va = coeffs[2];
  auto const vw = vc - p;
  Float const a = 2 * squaredNorm(va);
  Float const b = 3 * va.dot(vb);
  Float const c = squaredNorm(vb) + 2 * va.dot(vw);
  Float const d = vb.dot(vw);

  // Return the real part of the 3 roots of the cubic equation
  auto const roots = solveCubic(a, b, c, d);

  // Find the root which minimizes the squared distance to the point.
  // If the closest root is less than 0 or greater than 1, it isn't valid.
  // It's not clear that we can simply clamp the root to [0, 1], so we test
  // against v[0] and v[1] explicitly.

  Float r = 0;
  Float sq_dist = p.squaredDistanceTo(_v[0]);
  Float const sq_dist1 = p.squaredDistanceTo(_v[1]);
  if (sq_dist1 < sq_dist) {
    r = 1;
    sq_dist = sq_dist1;
  }

  for (auto const rr : roots) {
    if (0 <= rr && rr <= 1) {
      auto const p_root = (*this)(rr);
      Float const p_sq_dist = p.squaredDistanceTo(p_root);
      if (p_sq_dist < sq_dist) {
        r = rr;
        sq_dist = p_sq_dist;
      }
    }
  }
  return r;
}

//==============================================================================
// isStraight
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D> const & q) noexcept -> bool
{
  // Q(r) = C + rB + r²A,
  // if A is small, then the segment is straight.
  // Drop the multiplication by 2
  Point<D> const a = q[0] + q[1] - 2 * q[2];
  // a is effectively the 2 * the displacement of the q[2] from the midpoint of
  // q[0] and q[1]
  //
  // Note: we can't just do LineSegment(q[0], q[1]).distanceTo(q[2]) < eps_distance
  // The curve would not be straight if q[2] were very close to q[0] or q[1],
  // despite passing the distance test.
  return squaredNorm(a) < 16 * eps_distance2;
}

//==============================================================================
// enclosedArea
//==============================================================================

PURE HOSTDEV constexpr auto
enclosedArea(QuadraticSegment2 const & q) noexcept -> Float
{
  // The area bounded by the segment and the line between the endpoints is
  // 4/3 of the area of the triangle formed by the vertices.
  // Assumes that the segment is convex.
  Float constexpr two_thirds = static_cast<Float>(2) / static_cast<Float>(3);
  Vec2F const v02 = q[2] - q[0];
  Vec2F const v01 = q[1] - q[0];
  return two_thirds * v02.cross(v01);
}

//==============================================================================
// enclosedCentroid
//==============================================================================

PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2 const & q) noexcept -> Vec2F
{
  // For a quadratic segment, with P₁ = (0, 0), P₂ = (x₂, 0), and P₃ = (x₃, y₃),
  // if the area bounded by q and the x-axis is convex, it can be
  // shown that the centroid of the area bounded by the segment and x-axis
  // is given by
  // C = (3x₂ + 4x₃, 4y₃) / 10
  //
  // To find the centroid of the area bounded by the segment for a general
  // quadratic segment, we transform the segment so that P₁ = (0, 0),
  // then use a change of basis (rotation) from the standard basis to the
  // following basis, to achieve y₂ = 0.
  //
  // Let v = (v₁, v₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // u₁ = ( v₁,  v₂) = v
  // u₂ = (-v₂,  v₁)
  //
  // Note: u₁ and u₂ are orthonormal.
  //
  // The transformation from the new basis to the standard basis is given by
  // U = [u₁ u₂] = | v₁ -v₂ |
  //               | v₂  v₁ |
  //
  // Since u₁ and u₂ are orthonormal, U is unitary.
  //
  // The transformation from the standard basis to the new basis is given by
  // U⁻¹ = Uᵗ = |  v₁  v₂ |
  //            | -v₂  v₁ |
  // since U is unitary.
  //
  // To transfrom the segment to the x-axis, for each point P,
  // Pᵤ = Uᵗ * (P - P₁)
  // or
  // P₁ᵤ = (0, 0)
  // P₂ᵤ = Uᵗ * (P₂ - P₁)
  // P₃ᵤ = Uᵗ * (P₃ - P₁)
  //
  // P₂ᵤx = u₁ ⋅ (P₂ - P₁)
  // P₃ᵤx = u₁ ⋅ (P₃ - P₁)
  // P₃ᵤy = u₂ ⋅ (P₃ - P₁)
  // Hence, the centroid after the transformation is given by
  // Cᵤ = (3P₂ᵤx + 4P₃ᵤx, 4P₃ᵤy) / 10
  //
  // To transform the centroid back to the standard basis, we use
  // C = U * Cᵤ + P₁
  Vec2F const v12 = q[1] - q[0];
  Vec2F const v13 = q[2] - q[0];
  Vec2F const u1 = v12.normalized();
  Vec2F const u2(-u1[1], u1[0]);
  Mat2x2F const u(u1, u2);
  Float const p2ux = u1.dot(v12);
  Float const p3ux = u1.dot(v13);
  Float const p3uy = u2.dot(v13);
  Vec2F cu(3 * p2ux + 4 * p3ux, 4 * p3uy);
  Float constexpr tenth = static_cast<Float>(1) / static_cast<Float>(10);
  cu *= tenth;
  return u * cu + q[0];
}

//==============================================================================
// distanceTo
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::squaredDistanceTo(Vertex const & p) const noexcept -> Float
{
  Float const r = pointClosestTo(p);
  Vertex const p_closest = (*this)(r);
  return p_closest.squaredDistanceTo(p);
}

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::distanceTo(Vertex const & p) const noexcept -> Float
{
  return um2::sqrt(squaredDistanceTo(p));
}

//==============================================================================
// intersect
//==============================================================================

// The ray: R(r) = O + rD
// The quadratic segment: Q(s) = C + sB + s²A,
// where
//  C = P₁
//  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
//  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
// and
// V₁₃ = q[3] - q[1]
// V₂₃ = q[3] - q[2]
//
// O + rD = C + sB + s²A                          subtracting C from both sides
// rD = (C - O) + sB + s²A                        cross product with D (distributive)
// 0 = (C - O) × D + s(B × D) + s²(A × D)
// The cross product of two vectors in the plane is a vector of the form (0, 0, k).
// Let a = (A × D)ₖ, b = (B × D)ₖ, and c = ((C - O) × D)ₖ
// 0 = as² + bs + c
// If a = 0
//   s = -c/b
// else
//   s = (-b ± √(b²-4ac))/2a
// s is invalid if b² < 4ac
// Once we have a valid s
// P = Q(s) = C + sB + s²A
// O + rD = P ⟹   rD = (C - O) + s(B + sA)
//            ⟹   r = ([(C - O) + s(B + sA)] ⋅ D)/(D ⋅ D)
// Since D is a unit vector, we can simplify the above expression to
// r = [(C - O) + s(B + sA)] ⋅ D
// r is valid if 0 ≤ r ≤ ∞

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::intersect(Ray2 const ray) const noexcept -> Vec2F
requires(D == 2)
{
  //Vec2F const v01 = q[1] - q[0];
  Vec2F const v02 = _v[2] - _v[0];
  Vec2F const v12 = _v[2] - _v[1];

  Vec2F const va = -2 * (v02 + v12); // -2(V₁₃ + V₂₃)
  Vec2F const vb = 3 * v02 + v12;    // 3V₁₃ + V₂₃
  Vec2F const vc = _v[0];

  Vec2F const vd = ray.direction();
  Vec2F const vo = ray.origin();

  Vec2F const voc = vc - vo; // (C - O)

  Float const a = va.cross(vd);   // (A × D)ₖ
  Float const b = vb.cross(vd);   // (B × D)ₖ
  Float const c = voc.cross(vd); // ((C - O) × D)ₖ

  Vec2F result(inf_distance, inf_distance);
  auto constexpr epsilon = castIfNot<Float>(1e-8);
  if (um2::abs(a) < epsilon) {
    Float const s = -c / b;
    if (0 <= s && s <= 1) {
      Float const r = (voc + s * (vb + s * va)).dot(vd);
      if (0 <= r) {
        result[0] = r;
      }
    }
    return result;
  }
  Float const disc = b * b - 4 * a * c;
  if (disc < 0) {
    return result;
  }

  Float const s1 = (-b - um2::sqrt(disc)) / (2 * a);
  Float const s2 = (-b + um2::sqrt(disc)) / (2 * a);
  if (0 <= s1 && s1 <= 1) {
    Float const r = (voc + s1 * (vb + s1 * va)).dot(vd);
    if (0 <= r) {
      result[0] = r;
    }
  }
  if (0 <= s2 && s2 <= 1) {
    Float const r = (voc + s2 * (vb + s2 * va)).dot(vd);
    if (0 <= r) {
      result[1] = r;
    }
  }
  return result;
}

//==============================================================================
// curvesLeft
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::curvesLeft() const noexcept -> bool
requires (D == 2)
{
  return areCCW(_v[0], _v[2], _v[1]);
}

//==============================================================================
// getPolyCoeffs
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::getPolyCoeffs() const noexcept -> Vec<3, Vertex>
{
  // For quadratic segments, the parametric equation is
  //  Q(r) = P₁ + rB + r²A,
  // where
  //  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  //  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // and
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  Vec<D, Float> const v13 = _v[2] - _v[0];
  Vec<D, Float> const v23 = _v[2] - _v[1];
  Vec<D, Float> const b = 3 * v13 + v23;
  Vec<D, Float> const a = -2 * (v13 + v23);
  return {_v[0], b, a};
}

} // namespace um2
