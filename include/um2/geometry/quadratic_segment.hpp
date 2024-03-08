#pragma once

#include <um2/geometry/polytope.hpp>
#include <um2/geometry/line_segment.hpp>
#include <um2/stdlib/math.hpp>

// Need complex for quadratic segments
#if UM2_USE_CUDA
#  include <cuda/std/complex>
#else
#  include <complex>
#endif

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
  // This routine has previously been a major bottleneck, so some readability
  // has been sacrificed for performance. Hopefully the extensive comments
  // will make up for this.
  //
  // The quadratic segment has an equivalent quadratic Bezier curve.
  // The triangle formed by the control points of the quadratic Bezier curve
  // bound the curve. If p is not in this triangle we can ignore the curvature of the
  // segment and return based upon areCCW(v[0], v[1], p). If the triangle is CCW oriented,
  //    We will check that p is in the triangle by checking that p is left of each edge of
  //    the triangle.
  // else
  //    We will check that p is in the triangle by checking that p is right of each edge
  //    of the triangle.
  // We manually perform the check for (v0, v1) since we want to reuse v01 and v0p.
  Vec2F const v01 = _v[1] - _v[0];
  Vec2F const bcp = getBezierControlPoint(*this);
  Vec2F const v0b = bcp - _v[0];
  Vec2F const v0p = p - _v[0];
  bool const tri_is_ccw = v01.cross(v0b) >= 0;
  {
    bool const b0 = v01.cross(v0p) >= 0;  // areCCW(v[0], v[1], p) == Left of edge 0
    bool const b1 = areCCW(_v[1], bcp, p); // Left of edge 1
    bool const b2 = areCCW(bcp, _v[0], p); // Left of edge 2
    // if b0 && b1 && b2, the point is in the triangle, and we must perform further
    // analysis.
    // if !(b0 && b1 && b2), the point is outside the triangle, and we may return
    // b0
    bool const p_in_tri = tri_is_ccw ? (b0 && b1 && b2) : (!b0 && !b1 && !b2);
    if (!p_in_tri) {
      return b0;
    }
  }

  // We want to rotate the segment so that v[0] is at the origin and v[1] is on the
  // x-axis. We then take note of the sign of the rotated v[2] to determine if the
  // segment curves left or right. We will orient the segment so that it curves right.
  //     Compute the rotation matrix
  Float const v01_norm = v01.norm();
  //     We can avoid a matrix multiplication by using the fact that the y-coordinate of
  //     v1_r is zero.
  Vec2F const v1_r(v01_norm, static_cast<Float>(0));
  Vec2F const v01_normalized = v01 / v01_norm;
  //     NOLINTBEGIN(readability-identifier-naming) matrix notation
  Mat2x2F const R(Vec2F(v01_normalized[0], -v01_normalized[1]),
                  Vec2F(v01_normalized[1], v01_normalized[0]));
  Vec2F const v02 = _v[2] - _v[0];
  Vec2F v2_r = R * v02;
  Vec2F p_r = R * (p - _v[0]);
  bool const curves_right = v2_r[1] >= 0;
  if (!curves_right) {
    // Flip the y-coordinates to be greater than or equal to zero
    v2_r[1] = -v2_r[1];
    p_r[1] = -p_r[1];
  }
  // If v2_r[1] < epsilonDistance, then the segment is straight and we can use the cross
  // product test to return early.
  bool const is_straight = v2_r[1] <= eps_distance;
  if (is_straight) {
    // if !curves_right, we flipped the y-coordinates, so we need to flip the sign of the
    // cross product.
    // | positive cross product | curves_right | final result |
    // |------------------------|--------------|--------------|
    // | true                   | true         | true         |
    // | false                  | true         | false        |
    // | true                   | false        | false        |
    // | false                  | false        | true         |
    return (v1_r.cross(p_r) >= 0) ? curves_right : !curves_right;
  }

  // We now wish to compute the bounding box of the segment in order to determine if we
  // need to treat the segment as a straight line or a curve. If the point is outside the
  // triangle, then we can treat the segment like a straight line.
  //  Q(r) = C + rB + r²A
  // where
  //  C = (0, 0)
  //  B =  4 * v2_r - v1_r
  //  A = -4 * v2_r + 2 * v1_r = -B + v1_r
  // Q′(r) = B + 2rA,
  // The stationary points are (r_i,...) = -B / (2A)
  // Q(r_i) = -B² / (4A) = r_i(B/2)
  // Since v1_r[1] = 0 and v2_r[1] > 0, we know that:
  //    B[1] > 0
  //    A[1] = -B[1], which implies A[1] < 0
  //    A[1] < 0 implies Q(r_i) is strictly positive

  // If the point is below the origin, then the point is right of the segment
  if (p_r[1] < 0) {
    return !curves_right;
  }
  Float const Bx = 4 * v2_r[0] - v1_r[0];
  Float const By = 4 * v2_r[1]; // Positive
  Float const Ax = -Bx + v1_r[0];
  Float const Ay = -By; // Negative
  ASSERT_ASSUME(By > 0);
  ASSERT_ASSUME(Ay < 0);

  // If the point is in the bounding triangle of the segment,
  // we will find the point on the segment that shares the same x-coordinate
  //  Q(r) = C + rB + r²A = P
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A[0] == 0, then we have a well-behaved quadratic segment, and there is one root.
  // This is the expected case.
  if (um2::abs(Ax) < 4 * eps_distance) {
    // This is a linear equation, so there is only one root.
    Float const r = p_r[0] / Bx; // B[0] != 0, otherwise the segment would be degenerate.
    // We know the point is in the AABB of the segment, so we expect r to be in [0, 1]
    ASSERT_ASSUME(0 <= r);
    ASSERT_ASSUME(r <= 1);
    Float const Q_y = r * (By + r * Ay);
    // if p_r < Q_y, then the point is to the right of the segment
    return (p_r[1] <= Q_y) ? !curves_right : curves_right;
  }
  // Two roots.
  Float const disc = Bx * Bx + 4 * Ax * p_r[0];
  ASSERT_ASSUME(disc >= 0);
  Float const r1 = (-Bx + um2::sqrt(disc)) / (2 * Ax);
  Float const r2 = (-Bx - um2::sqrt(disc)) / (2 * Ax);
  Float const Q_y1 = r1 * (By + r1 * Ay);
  Float const Q_y2 = r2 * (By + r2 * Ay);
  Float const Q_ymin = um2::min(Q_y1, Q_y2);
  Float const Q_ymax = um2::max(Q_y1, Q_y2);
  bool const contained_in_curve = Q_ymin <= p_r[1] && p_r[1] <= Q_ymax;
  return contained_in_curve ? !curves_right : curves_right;
  // NOLINTEND(readability-identifier-naming)
}

//==============================================================================
// length
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::length() const noexcept -> Float
{
  // Turn off variable naming convention warning for this function, since we will use
  // capital letters to denote vectors.
  // NOLINTBEGIN(readability-identifier-naming) mathematical convention

  // The arc length integral may be reduced to an integral over the square root of a
  // quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
  //              1             1
  // arc length = ∫ ‖Q′(r)‖dr = ∫ √(ar² + br + c) dr
  //              0             0
  //
  // If a = 0, we need to use a different formula, else the result is NaN.
  //  Q(r) = C + rB + r²A,
  // where
  //  C = P₁
  //  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  //  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // and
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  // Q′(r) = B + 2rA,

  if (isStraight(*this)) {
    return _v[0].distanceTo(_v[1]);
  }

  Vec<D, Float> const v13 = _v[2] - _v[0];
  Vec<D, Float> const v23 = _v[2] - _v[1];
  Vec<D, Float> const A = -2 * (v13 + v23);

  // ‖Q′(r)‖ =  √(4(A ⋅A)r² + 4(A ⋅B)r + B ⋅B) = √(ar² + br + c)
  // where
  // a = 4(A ⋅ A)
  // b = 4(A ⋅ B)
  // c = B ⋅ B

  Vec<D, Float> const B = 3 * v13 + v23;
  ASSERT(squaredNorm(A) > eps_distance2);
  Float const a = 4 * squaredNorm(A);
  Float const b = 4 * dot(A, B);
  Float const c = squaredNorm(B);

  // √(ar² + br + c) = √a √( (r + b₁)^2 + c₁)
  // where
  Float const b1 = b / (2 * a);
  Float const c1 = (c / a) - (b1 * b1);
  // The step above with division by a is safe, since a ≠ 0.

  // Let u = r + b₁, then
  // 1                       1 + b₁
  // ∫ √(ar² + br + c) dr = √a ∫ √(u² + c₁) du
  // 0                         b₁
  //
  // This is an integral that exists in common integral tables.
  // Evaluation of the resultant expression may be simplified by using

  Float const lb = b1;
  Float const ub = 1 + b1;
  Float const L = um2::sqrt(c1 + lb * lb);
  Float const U = um2::sqrt(c1 + ub * ub);
  // Numerical issues may cause the bounds to be slightly outside the range [-1, 1].
  // If we go too far outside this range, error out as something has gone wrong.
  auto constexpr epsilon = castIfNot<Float>(1e-5);
  Float const lb_l = lb / L;
  Float const ub_u = ub / U;
  Float constexpr clamp_bound = static_cast<Float>(1) - epsilon;
#if UM2_ENABLE_ASSERTS
  Float constexpr assert_bound = static_cast<Float>(1) + epsilon;
  ASSERT(-assert_bound <= lb_l);
  ASSERT(lb_l <= assert_bound);
  ASSERT(-assert_bound <= ub_u);
  ASSERT(ub_u <= assert_bound);
#endif
  Float const arg_l = um2::clamp(lb_l, -clamp_bound, clamp_bound);
  Float const arg_u = um2::clamp(ub_u, -clamp_bound, clamp_bound);
  Float const atanh_l = um2::atanh(arg_l);
  Float const atanh_u = um2::atanh(arg_u);
  Float const result = um2::sqrt(a) * (U + lb * (U - L) + c1 * (atanh_u - atanh_l)) / 2;
  ASSERT(0 <= result);
  ASSERT(result <= inf_distance);
  return result;
  // NOLINTEND(readability-identifier-naming)
}

//==============================================================================
// boundingBox
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  // Floatind the extrema by finding dx_i/dr = 0
  //  Q(r) = P₁ + rB + r²A,
  // where
  //  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  //  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // and
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  // Q′(r) = B + 2rA,
  // (r_i,...) = -B / (2A)
  // x_i = Q(r_i) = P₁ - B² / (4A)
  // Compare the extrema with the segment's endpoints to find the AABox
  Vec<D, Float> const v02 = _v[2] - _v[0];
  Vec<D, Float> const v12 = _v[2] - _v[1];
  Point<D> minima = um2::min(_v[0], _v[1]);
  Point<D> maxima = um2::max(_v[0], _v[1]);
  for (Int i = 0; i < D; ++i) {
    Float const a = -2 * (v02[i] + v12[i]);
    if (um2::abs(a) < 4 * eps_distance) {
      // The segment is almost a straight line, so the extrema are the endpoints.
      continue;
    }
    // r_i = -B_i / (2A_i)
    Float const half_b = (3 * v02[i] + v12[i]) / 2;
    Float const r = -half_b / a;
    // if r is not in [0, 1], then the extrema are not on the segment, hence
    // the segment's endpoints are the extrema.
    if (0 < r && r < 1) {
      // x_i = Q(r_i) = P₁ - B² / (4A) = P₁ + r(B/2)
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

// NOLINTBEGIN(readability-identifier-naming) Mathematical notation
template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::pointClosestTo(Vertex const & p) const noexcept -> Float
{
  // We want to use the complex functions in the std or cuda::std namespace
  // depending on if we're using CUDA
  // NOLINTBEGIN(google-build-using-namespace)
#if UM2_USE_CUDA
  namespace math = cuda::std;
#else
  namespace math = std;
#endif
  // NOLINTEND(google-build-using-namespace)

  if (isStraight(*this)) {
    return LineSegment<D>(_v[0], _v[1]).pointClosestTo(p);
  }

  // Note the 1-based indexing in this section
  //
  // The interpolation function of the quadratic segment is
  // Q(r) = C + rB + r²A,
  // where
  // C = P₁
  // B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  // A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  //
  // We wish to find r which minimizes ‖P - Q(r)‖.
  // This r also minimizes ‖P - Q(r)‖².
  // It can be shown that this is equivalent to finding the minimum of the
  // quartic function
  // ‖P - Q(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
  // Let W = P - P₁ = P - C
  // a₄ = A ⋅ A
  // a₃ = 2(A ⋅ B)
  // a₂ = -2(A ⋅ W) + (B ⋅ B)
  // a₁ = -2(B ⋅ W)
  // a₀ = W ⋅ W
  //
  // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
  // a = 2(A ⋅ A)
  // b = 3(A ⋅ B)
  // c = (B ⋅ B) - 2(A ⋅W)
  // d = -(B ⋅ W)
  // Note we factored out a 2
  //
  // We can then use Lagrange's method is used to find the roots.
  // (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
  Vec<D, Float> const v13 = _v[2] - _v[0];
  Vec<D, Float> const v23 = _v[2] - _v[1];
  Vec<D, Float> const A = -2 * (v13 + v23);
  Float const a = 2 * squaredNorm(A);
  Vec<D, Float> const B = 3 * v13 + v23;
  Float const b = 3 * dot(A, B);
  Vec<D, Float> const W = p - _v[0];
  Float const c = squaredNorm(B) - 2 * dot(A, W);
  Float const d = -dot(B, W);

  // Lagrange's method
  // Compute the elementary symmetric functions
  Float const e1 = -b / a; // Note for later s0 = e1
  Float const e2 = c / a;
  Float const e3 = -d / a;
  // Compute the symmetric functions
  Float const P = e1 * e1 - 3 * e2;
  Float const S = 2 * e1 * e1 * e1 - 9 * e1 * e2 + 27 * e3;
  // We solve z^2 - Sz + P^3 = 0
  Float const disc = S * S - 4 * P * P * P;
  auto const eps = castIfNot<Float>(1e-7);

  //  assert(um2::abs(disc) > eps); // 0 single or double root
  //  if (0 < disc) { // One real root
  //    Float const s1 = um2::cbrt((S + um2::sqrt(disc)) / 2);
  //    Float const s2 = (um2::abs(s1) < eps) ? 0 : P / s1;
  //    // Using s0 = e1
  //    return (e1 + s1 + s2) / 3;
  //  }
  // A complex cbrt
  Float constexpr half = static_cast<Float>(1) / 2;
  Float constexpr third = static_cast<Float>(1) / 3;
  math::complex<Float> const s1 = math::exp(
      math::log((S + math::sqrt(static_cast<math::complex<Float>>(disc))) * half) * third);
  math::complex<Float> const s2 = (abs(s1) < eps) ? 0 : P / s1;
  // zeta1 = (-1/2, sqrt(3)/2)
  math::complex<Float> const zeta1(-half, um2::sqrt(static_cast<Float>(3)) / 2);
  math::complex<Float> const zeta2(conj(zeta1));

  // Find the real root that minimizes the distance to p
  Float r = 0;
  Float dist = p.squaredDistanceTo(_v[0]);
  if (p.squaredDistanceTo(_v[1]) < dist) {
    r = 1;
    dist = p.squaredDistanceTo(_v[1]);
  }

  Vec3<Float> const rr((e1 + real(s1 + s2)) / 3, (e1 + real(zeta2 * s1 + zeta1 * s2)) / 3,
                   (e1 + real(zeta1 * s1 + zeta2 * s2)) / 3);
  for (Int i = 0; i < 3; ++i) {
    Float const rc = rr[i];
    if (0 <= rc && rc <= 1) {
      Float const dc = p.squaredDistanceTo((*this)(rc));
      if (dc < dist) {
        r = rc;
        dist = dc;
      }
    }
  }
  return r;
}
// NOLINTEND(readability-identifier-naming)

//==============================================================================
// isStraight
//==============================================================================

template <Int D>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D> const & q) noexcept -> bool
{
  // LineSegment l(v[0], v[1]);
  // Find the point p on l that is closest to v[2]
  // if p.squaredDistanceTo(v[2]) < eps_distance2, then we can treat the segment as
  // straight.
  LineSegment<D> const l(q[0], q[1]);
  Point<D> const p = l(l.pointClosestTo(q[2])); 
  return p.squaredDistanceTo(q[2]) < eps_distance2;
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
  // where 0 < x₂, if the area bounded by q and the x-axis is convex, it can be
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
  // Therefore, the centroid of the area bounded by the segment is given by
  // C = U * Cᵤ + P₁
  // where
  // Cᵤ = (u₁ ⋅ (3(P₂ - P₁) + 4(P₃ - P₁)), 4(u₂ ⋅ (P₃ - P₁))) / 10
  Vec2F const v12 = q[1] - q[0];
  Vec2F const four_v13 = 4 * (q[2] - q[0]);
  Vec2F const u1 = v12.normalized();
  Vec2F const u2(-u1[1], u1[0]);
  // NOLINTBEGIN(readability-identifier-naming) capitalize matrix
  Mat2x2F const U(u1, u2);
  Vec2F const Cu(u1.dot((3 * v12 + four_v13)) / 10, u2.dot(four_v13) / 10);
  return U * Cu + q[0];
  // NOLINTEND(readability-identifier-naming)
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
// O + rD = P ⟹   r = ((P - O) ⋅ D)/(D ⋅ D)

template <Int D>
PURE HOSTDEV constexpr auto
QuadraticSegment<D>::intersect(Ray2 const ray) const noexcept -> Vec2F
requires(D == 2)
{
  // NOLINTBEGIN(readability-identifier-naming) mathematical notation
  // This code is called very frequently so we sacrifice readability for speed.
  //Vec2F const v01 = q[1] - q[0];
  Vec2F const v02 = _v[2] - _v[0];
  Vec2F const v12 = _v[2] - _v[1];

  Vec2F const A = -2 * (v02 + v12); // -2(V₁₃ + V₂₃)
  Vec2F const B = 3 * v02 + v12;    // 3V₁₃ + V₂₃
  // Vec2F const C = q[0];

  // Vec2F const D = ray.d;
  // Vec2F const O = ray.o;

  Vec2F const voc = _v[0] - ray.origin(); // C - O

  Float const a = A.cross(ray.direction());   // (A × D)ₖ
  Float const b = B.cross(ray.direction());   // (B × D)ₖ
  Float const c = voc.cross(ray.direction()); // ((C - O) × D)ₖ

  Vec2F result(inf_distance, inf_distance);
  auto constexpr epsilon = castIfNot<Float>(1e-8);
  if (um2::abs(a) < epsilon) {
    Float const s = -c / b;
    if (0 <= s && s <= 1) {
      Vec2F const P = s * (s * A + B) + voc;
      Float const r0 = dot(P, ray.direction());
      // ray direction is a unit vector, no need for / ray.direction().squaredNorm();
      if (0 <= r0) {
        result[0] = r0;
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
    Vec2F const P = s1 * (s1 * A + B) + voc;
    Float const r0 = dot(P, ray.direction());
    // / ray.direction().squaredNorm();
    if (0 <= r0) {
      result[0] = r0;
    }
  }
  if (0 <= s2 && s2 <= 1) {
    Vec2F const P = s2 * (s2 * A + B) + voc;
    Float const r1 = dot(P, ray.direction());
    /// ray.direction().squaredNorm();
    if (0 <= r1) {
      result[1] = r1;
    }
  }
  // NOLINTEND(readability-identifier-naming)
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

} // namespace um2
