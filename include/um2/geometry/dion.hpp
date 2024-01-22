#pragma once

#include <um2/geometry/polytope.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/mat.hpp>

// Need complex for quadratic segments
#if UM2_USE_CUDA
#  include <cuda/std/complex>
#else
#  include <complex>
#endif

//==============================================================================
// DION
//==============================================================================
// A 1-dimensional polytope, of polynomial order P, represented by the connectivity
// of its vertices. These N vertices are D-dimensional points of type F.
//
// For Dions:
//   LineSegment (P = 1, N = 2)
//   QuadraticSegment (P = 2, N = 3)
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

template <Size P, Size N, Size D>
class Polytope<1, P, N, D>
{

public:
  using Vertex = Point<D>; 

private:
  Vertex _v[N];

public:
  //==============================================================================
  // Accessors
  //==============================================================================

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> Vertex &;

  // Returns the i-th vertex
  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> Vertex const &;

  // Returns a pointer to the vertex array
  PURE HOSTDEV [[nodiscard]] constexpr auto
  vertices() const noexcept -> Vertex const *;

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Polytope() noexcept = default;

  template <class... Pts>
    requires(sizeof...(Pts) == N && (std::same_as<Vertex, Pts> && ...))
  // NOLINTNEXTLINE(google-explicit-constructor) implicit conversion is desired
  HOSTDEV constexpr Polytope(Pts const... args) noexcept
      : _v{args...}
  {
  }

  //==============================================================================
  // Methods
  //==============================================================================

  // Interpolate along the segment.
  // r in [0, 1] are valid values.
  // F(r) -> (x, y, z)
  template <typename R>
  PURE HOSTDEV constexpr auto
  operator()(R r) const noexcept -> Vertex;

  // dF/dr (r) -> (dx/dr, dy/dr, dz/dr)
  template <typename R>
  PURE HOSTDEV [[nodiscard]] constexpr auto
  jacobian(R r) const noexcept -> Vec<D, F>;

  // Get the matrix that rotates the polytope such that the first and l
  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // The rotation matrix is returned.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRotation() const noexcept -> Mat<D, D, F>
    requires(D == 2);

  // If a point is to the left of the segment, with the segment oriented from
  // r = 0 to r = 1.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLeft(Vertex const & p) const noexcept -> bool
    requires(D == 2);

  // Arc length of the segment
  PURE HOSTDEV [[nodiscard]] constexpr auto
  length() const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  boundingBox() const noexcept -> AxisAlignedBox<D>;

  // Return the point on the curve that is closest to the given point.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  pointClosestTo(Vertex const & p) const noexcept -> F;

  // Return the squared distance between the given point and the closest point
  // on the curve.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  squaredDistanceTo(Vertex const & p) const noexcept -> F;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  distanceTo(Vertex const & p) const noexcept -> F;

}; // Dion

//==============================================================================
// Accessors
//==============================================================================

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::operator[](Size i) noexcept -> Vertex &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::operator[](Size i) const noexcept -> Vertex const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT_ASSUME(i < N);
  return _v[i];
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::vertices() const noexcept -> Vertex const *
{
  return _v;
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename R>
PURE HOSTDEV constexpr auto
interpolate(LineSegment<D> const & l, R const r) noexcept -> Point<D>
{
  // L(r) = (1 - r) * v0 + r * v1
  F const rr = static_cast<F>(r);
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = l[0][i] + rr * (l[1][i] - l[0][i]);
  }
  return result;
}

template <Size D, typename R>
PURE HOSTDEV constexpr auto
interpolate(QuadraticSegment<D> const & q, R const r) noexcept -> Point<D>
{
  // Q(r) =
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  F const rr = static_cast<F>(r);
  F const two_rr_1 = 2 * rr - 1;
  F const rr_1 = rr - 1;

  F const w0 = two_rr_1 * rr_1;
  F const w1 = two_rr_1 * rr;
  F const w2 = -4 * rr * rr_1;
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * q[0][i] + w1 * q[1][i] + w2 * q[2][i];
  }
  return result;
}

template <Size P, Size N, Size D>
template <typename R>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::operator()(R const r) const noexcept -> Vertex
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename R>
PURE HOSTDEV constexpr auto
jacobian(LineSegment<D> const & l, R const /*r*/) noexcept -> Vec<D, F>
{
  return l[1] - l[0];
}

template <Size D, typename R>
PURE HOSTDEV constexpr auto
jacobian(QuadraticSegment<D> const & q, R const r) noexcept -> Vec<D, F>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  F const w0 = 4 * static_cast<F>(r) - 3;
  F const w1 = 4 * static_cast<F>(r) - 1;
  Vec<D, F> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * (q[0][i] - q[2][i]) + w1 * (q[1][i] - q[2][i]);
  }
  return result;
}

template <Size P, Size N, Size D>
template <typename R>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::jacobian(R const r) const noexcept -> Vec<D, F>
{
  return um2::jacobian(*this, r);
}

//==============================================================================
// getRotation
//==============================================================================

PURE HOSTDEV constexpr auto
getRotation(LineSegment2 const & l) noexcept -> Mat2x2<F>
{
  // We want to transform the segment so that v[0] is at the origin and v[1]
  // is on the x-axis. We can do this by first translating by -v[0] and then
  // using a change of basis (rotation) matrix to rotate v[1] onto the x-axis.
  // x_old = U * x_new
  //
  // For 2D:
  // Let a = (a₁, a₂) = (P₂ - P₁) / ‖P₂ - P₁‖
  // u₁ = ( a₁,  a₂) = a
  // u₂ = (-a₂,  a₁)
  //
  // Note: u₁ and u₂ are orthonormal.
  //
  // The transformation from the new basis to the standard basis is given by
  // U = [u₁ u₂] = | a₁ -a₂ |
  //               | a₂  a₁ |
  //
  // Since u₁ and u₂ are orthonormal, U is unitary.
  //
  // The transformation from the standard basis to the new basis is given by
  // U⁻¹ = Uᵗ = |  a₁  a₂ |
  //            | -a₂  a₁ |
  // since U is unitary.
  Vec2<F> const a = (l[1] - l[0]).normalized();
  Vec2<F> const col0(a[0], -a[1]);
  Vec2<F> const col1(a[1], a[0]);
  return Mat2x2<F>(col0, col1);
}

PURE HOSTDEV constexpr auto
getRotation(QuadraticSegment2 const & q) noexcept -> Mat2x2<F>
{
  return LineSegment2(q[0], q[1]).getRotation();
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::getRotation() const noexcept -> Mat<D, D, F>
  requires(D == 2)
{
  return um2::getRotation(*this);
}

//==============================================================================
// getBezierControlPoint
//==============================================================================
// Get the control point for the quadratic Bezier curve that interpolates the
// given quadratic segment.

template <Size D>
PURE HOSTDEV constexpr auto
getBezierControlPoint(QuadraticSegment<D> const & q) noexcept -> Point<D>
{
  // p0 == v[0]
  // p2 == v[1]
  // p1 == 2 * v[2] - (v[0] + v[1]) / 2, hence we only need to compute p1
  Point<D> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = static_cast<F>(2) * q[2][i] - (q[0][i] + q[1][i]) / 2;
  }
  return result;
}

//==============================================================================
// pointIsLeft
//==============================================================================

PURE HOSTDEV constexpr auto
pointIsLeft(LineSegment2 const & l, Point2 const & p) noexcept -> bool
{
  return areCCW(l[0], l[1], p);
}

PURE HOSTDEV constexpr auto
pointIsLeft(QuadraticSegment2 const & q, Point2 const & p) noexcept -> bool
{
  // This routine has previously been a major bottleneck, so some readability
  // has been sacrificed for performance.
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
  Vec2<F> const v01(q[1][0] - q[0][0], q[1][1] - q[0][1]);
  Point2 const bcp = getBezierControlPoint(q);
  Vec2<F> const v0b(bcp[0] - q[0][0], bcp[1] - q[0][1]);
  Vec2<F> const v0p(p[0] - q[0][0], p[1] - q[0][1]);
  bool const tri_is_ccw = v01.cross(v0b) >= 0;
  {
    bool const b0 = v01.cross(v0p) >= 0;  // areCCW(v[0], v[1], p) == Left of edge 0
    bool const b1 = areCCW(q[1], bcp, p); // Left of edge 1
    bool const b2 = areCCW(bcp, q[0], p); // Left of edge 2
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
  F const v01_norm = v01.norm();
  //     We can avoid a matrix multiplication by using the fact that the y-coordinate of
  //     v1_r is zero.
  Point2 const v1_r(v01_norm, static_cast<F>(0));
  Vec2<F> const v01_normalized = v01 / v01_norm;
  //     NOLINTBEGIN(readability-identifier-naming) matrix notation
  Mat2x2<F> const R(Vec2<F>(v01_normalized[0], -v01_normalized[1]),
                 Vec2<F>(v01_normalized[1], v01_normalized[0]));
  Vec2<F> const v02 = q[2] - q[0];
  Point2 v2_r = R * v02;
  Point2 p_r = R * (p - q[0]);
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
  F const Bx = 4 * v2_r[0] - v1_r[0];
  F const By = 4 * v2_r[1]; // Positive
  F const Ax = -Bx + v1_r[0];
  F const Ay = -By; // Negative
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
    F const r = p_r[0] / Bx; // B[0] != 0, otherwise the segment would be degenerate.
    // We know the point is in the AABB of the segment, so we expect r to be in [0, 1]
    ASSERT_ASSUME(0 <= r);
    ASSERT_ASSUME(r <= 1);
    F const Q_y = r * (By + r * Ay);
    // if p_r < Q_y, then the point is to the right of the segment
    return (p_r[1] <= Q_y) ? !curves_right : curves_right;
  }
  // Two roots.
  F const disc = Bx * Bx + 4 * Ax * p_r[0];
  ASSERT_ASSUME(disc >= 0);
  F const r1 = (-Bx + um2::sqrt(disc)) / (2 * Ax);
  F const r2 = (-Bx - um2::sqrt(disc)) / (2 * Ax);
  F const Q_y1 = r1 * (By + r1 * Ay);
  F const Q_y2 = r2 * (By + r2 * Ay);
  F const Q_ymin = um2::min(Q_y1, Q_y2);
  F const Q_ymax = um2::max(Q_y1, Q_y2);
  bool const contained_in_curve = Q_ymin <= p_r[1] && p_r[1] <= Q_ymax;
  return contained_in_curve ? !curves_right : curves_right;
  // NOLINTEND(readability-identifier-naming)
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::isLeft(Vertex const & p) const noexcept -> bool
  requires(D == 2)
{
  return pointIsLeft(*this, p);
}

//==============================================================================
// length
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
length(LineSegment<D> const & l) noexcept -> F
{
  return (l[0]).distanceTo(l[1]);
}

template <Size D>
PURE HOSTDEV constexpr auto
length(QuadraticSegment<D> const & q) noexcept -> F
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

  if (isStraight(q)) {
    return q[0].distanceTo(q[1]);
  }

  Vec<D, F> const v13 = q[2] - q[0];
  Vec<D, F> const v23 = q[2] - q[1];
  Vec<D, F> A;
  for (Size i = 0; i < D; ++i) {
    A[i] = -2 * (v13[i] + v23[i]);
  }

  // ‖Q′(r)‖ =  √(4(A ⋅A)r² + 4(A ⋅B)r + B ⋅B) = √(ar² + br + c)
  // where
  // a = 4(A ⋅ A)
  // b = 4(A ⋅ B)
  // c = B ⋅ B

  Vec<D, F> B;
  for (Size i = 0; i < D; ++i) {
    B[i] = 3 * v13[i] + v23[i];
  }
  ASSERT(squaredNorm(A) > eps_distance2);
  F const a = 4 * squaredNorm(A);
  F const b = 4 * dot(A, B);
  F const c = squaredNorm(B);

  // √(ar² + br + c) = √a √( (r + b₁)^2 + c₁)
  // where
  F const b1 = b / (2 * a);
  F const c1 = (c / a) - (b1 * b1);
  // The step above with division by a is safe, since a ≠ 0.

  // Let u = r + b₁, then
  // 1                       1 + b₁
  // ∫ √(ar² + br + c) dr = √a ∫ √(u² + c₁) du
  // 0                         b₁
  //
  // This is an integral that exists in common integral tables.
  // Evaluation of the resultant expression may be simplified by using

  F const lb = b1;
  F const ub = 1 + b1;
  F const L = um2::sqrt(c1 + lb * lb);
  F const U = um2::sqrt(c1 + ub * ub);
  // Numerical issues may cause the bounds to be slightly outside the range [-1, 1].
  // If we go too far outside this range, error out as something has gone wrong.
#if UM2_ENABLE_FLOAT64
  F constexpr epsilon = 1e-5;
#else
  F constexpr epsilon = 1e-5f;
#endif
  F const lb_l = lb / L; 
  F const ub_u = ub / U;
  F constexpr clamp_bound = static_cast<F>(1) - epsilon;
#if UM2_ENABLE_ASSERTS
  F constexpr assert_bound = static_cast<F>(1) + epsilon;
  ASSERT(-assert_bound <= lb_l);
  ASSERT(lb_l <= assert_bound);
  ASSERT(-assert_bound <= ub_u);
  ASSERT(ub_u <= assert_bound);
#endif
  F const arg_l = um2::clamp(lb_l, -clamp_bound, clamp_bound);
  F const arg_u = um2::clamp(ub_u, -clamp_bound, clamp_bound); 
  F const atanh_l = um2::atanh(arg_l);
  F const atanh_u = um2::atanh(arg_u);
  F const result = um2::sqrt(a) * (U + lb * (U - L) + c1 * (atanh_u - atanh_l)) / 2;
  ASSERT(0 <= result);
  ASSERT(result <= inf_distance);
  return result;
  // NOLINTEND(readability-identifier-naming)
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::length() const noexcept -> F
{
  return um2::length(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

// Defined in Polytope.hpp for the line segment, since for all linear polytopes
// the bounding box is simply the bounding box of the vertices.

template <Size D>
PURE HOSTDEV constexpr auto
boundingBox(QuadraticSegment<D> const & q) noexcept -> AxisAlignedBox<D>
{
  // Find the extrema by finding dx_i/dr = 0
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
  Vec<D, F> const v02 = q[2] - q[0];
  Vec<D, F> const v12 = q[2] - q[1];
  Point<D> minima = um2::min(q[0], q[1]);
  Point<D> maxima = um2::max(q[0], q[1]);
  for (Size i = 0; i < D; ++i) {
    F const a = -2 * (v02[i] + v12[i]);
    if (um2::abs(a) < 4 * eps_distance) {
      // The segment is almost a straight line, so the extrema are the endpoints.
      continue;
    }
    // r_i = -B_i / (2A_i)
    F const half_b = (3 * v02[i] + v12[i]) / 2;
    F const r = -half_b / a;
    // if r is not in [0, 1], then the extrema are not on the segment, hence
    // the segment's endpoints are the extrema.
    // NOLINTNEXTLINE(misc-redundant-expression) false positive
    if (0 < r && r < 1) {
      // x_i = Q(r_i) = P₁ - B² / (4A) = P₁ + r(B/2)
      F const x = q[0][i] + r * half_b;
      minima[i] = um2::min(minima[i], x);
      maxima[i] = um2::max(maxima[i], x);
    }
  }
  return AxisAlignedBox<D>{minima, maxima};
}

//==============================================================================
// pointClosestTo
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
pointClosestTo(LineSegment<D> const & l, Point<D> const & p) noexcept -> F
{
  // From Real-Time Collision Detection, Christer Ericson, 2005
  // Given segment ab and point c, computes closest point d on ab.
  // Returns t for the position of d, d(r) = a + r*(b - a)
  Vec<D, F> const ab = l[1] - l[0];
  // Project c onto ab, computing parameterized position d(r) = a + r*(b − a)
  F r = (p - l[0]).dot(ab) / ab.squaredNorm();
  // If outside segment, clamp r (and therefore d) to the closest endpoint
  if (r < 0) {
    r = 0;
  }
  if (r > 1) {
    r = 1;
  }
  return r;
}

// NOLINTBEGIN(readability-identifier-naming) Mathematical notation
template <Size D>
PURE HOSTDEV constexpr auto
pointClosestTo(QuadraticSegment<D> const & q, Point<D> const & p) noexcept -> F
{

  // We want to use the complex functions in the std or cuda::std namespace
  // depending on if we're using CUDA
  // NOLINTBEGIN(google-build-using-namespace)
#if UM2_USE_CUDA
  using namespace cuda::std;
#else
  using namespace std;
#endif
  // NOLINTEND(google-build-using-namespace)

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
  Vec<D, F> const v13 = q[2] - q[0];
  Vec<D, F> const v23 = q[2] - q[1];
  Vec<D, F> const A = -2 * (v13 + v23);
  F const a = 2 * squaredNorm(A);
  // 0 ≤ a, since a = 2(A ⋅ A)  = 2 ‖A‖², and 0 ≤ ‖A‖²
  // A = 4(midpoint of line - p3) -> a = 32 ‖midpoint of line - p3‖²
  // if a is small, then the segment is almost a straight line, and we should use the
  // line segment method
  if (a < 32 * eps_distance2) {
    Vec<D, F> const ab = q[1] - q[0];
    F r = (p - q[0]).dot(ab) / ab.squaredNorm();
    if (r < 0) {
      r = 0;
    }
    if (r > 1) {
      r = 1;
    }
    return r;
  }
  Vec<D, F> const B = 3 * v13 + v23;
  F const b = 3 * dot(A, B);
  Vec<D, F> const W = p - q[0];
  F const c = squaredNorm(B) - 2 * dot(A, W);
  F const d = -dot(B, W);

  // Lagrange's method
  // Compute the elementary symmetric functions
  F const e1 = -b / a; // Note for later s0 = e1
  F const e2 = c / a;
  F const e3 = -d / a;
  // Compute the symmetric functions
  F const P = e1 * e1 - 3 * e2;
  F const S = 2 * e1 * e1 * e1 - 9 * e1 * e2 + 27 * e3;
  // We solve z^2 - Sz + P^3 = 0
  F const disc = S * S - 4 * P * P * P;
#if UM2_ENABLE_FLOAT64
  F const eps = 1e-7;
#else
  F const eps = 1e-7f;
#endif

  //  assert(um2::abs(disc) > eps); // 0 single or double root
  //  if (0 < disc) { // One real root
  //    F const s1 = um2::cbrt((S + um2::sqrt(disc)) / 2);
  //    F const s2 = (um2::abs(s1) < eps) ? 0 : P / s1;
  //    // Using s0 = e1
  //    return (e1 + s1 + s2) / 3;
  //  }
  // A complex cbrt
  F constexpr half = static_cast<F>(1) / 2;
  F constexpr third = static_cast<F>(1) / 3;
  complex<F> const s1 =
      exp(log((S + sqrt(static_cast<complex<F>>(disc))) * half) * third);
  complex<F> const s2 = (abs(s1) < eps) ? 0 : P / s1;
  // zeta1 = (-1/2, sqrt(3)/2)
  complex<F> const zeta1(-half, um2::sqrt(static_cast<F>(3)) / 2);
  complex<F> const zeta2(conj(zeta1));

  // Find the real root that minimizes the distance to p
  F r = 0;
  F dist = p.squaredDistanceTo(q[0]);
  if (p.squaredDistanceTo(q[1]) < dist) {
    r = 1;
    dist = p.squaredDistanceTo(q[1]);
  }

  Vec3<F> const rr((e1 + real(s1 + s2)) / 3, (e1 + real(zeta2 * s1 + zeta1 * s2)) / 3,
                   (e1 + real(zeta1 * s1 + zeta2 * s2)) / 3);
  for (Size i = 0; i < 3; ++i) {
    F const rc = rr[i];
    if (0 <= rc && rc <= 1) {
      F const dc = p.squaredDistanceTo(q(rc));
      if (dc < dist) {
        r = rc;
        dist = dc;
      }
    }
  }
  return r;
}
// NOLINTEND(readability-identifier-naming)

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::pointClosestTo(Vertex const & p) const noexcept -> F
{
  return um2::pointClosestTo(*this, p);
}

//==============================================================================
// isStraight
//==============================================================================

template <Size D>
PURE HOSTDEV constexpr auto
isStraight(QuadraticSegment<D> const & q) noexcept -> bool
{
  // A slightly more optimized version of doing:
  // LineSegment(v[0], v[1]).distanceTo(v[2]) < epsilonDistance
  //
  // Compute the point on the line v[0] + r * (v[1] - v[0]) that is closest to v[2]
  Vec<D, F> const v01 = q[1] - q[0];
  F const r = (q[2] - q[0]).dot(v01) / v01.squaredNorm();
  // If r is outside the range [0, 1], then the segment is not straight
  if (r < 0 || r > 1) {
    return false;
  }
  // Compute the point on the line
  Vec<D, F> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = q[0][i] + r * v01[i];
  }
  // Check if the point is within epsilon distance of v[2]
  return isApprox(p, q[2]);
}

//==============================================================================
// enclosedArea
//==============================================================================

PURE HOSTDEV constexpr auto
enclosedArea(QuadraticSegment2 const & q) noexcept -> F
{
  // The area bounded by the segment and the line between the endpoints is
  // 4/3 of the area of the triangle formed by the vertices.
  // Assumes that the segment is convex.
  F constexpr two_thirds = static_cast<F>(2) / static_cast<F>(3);
  Vec2<F> const v02 = q[2] - q[0];
  Vec2<F> const v01 = q[1] - q[0];
  return two_thirds * v02.cross(v01);
}

//==============================================================================
// enclosedCentroid
//==============================================================================

PURE HOSTDEV constexpr auto
enclosedCentroid(QuadraticSegment2 const & q) noexcept -> Point2
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
  Vec2<F> const v12 = q[1] - q[0];
  Vec2<F> const four_v13 = 4 * (q[2] - q[0]);
  Vec2<F> const u1 = v12.normalized();
  Vec2<F> const u2(-u1[1], u1[0]);
  // NOLINTBEGIN(readability-identifier-naming) capitalize matrix
  Mat2x2<F> const U(u1, u2);
  Vec2<F> const Cu(u1.dot((3 * v12 + four_v13)) / 10, u2.dot(four_v13) / 10);
  return U * Cu + q[0];
  // NOLINTEND(readability-identifier-naming)
}

//==============================================================================
// intersect
//==============================================================================

// Returns the value r such that R(r) = L(s).
// If such a value does not exist, infiniteDistance is returned instead.
// 1) P₁ + s(P₂ - P₁) = O + rD           subtracting P₁ from both sides
// 2) s(P₂ - P₁) = (O - P₁) + rD         let U = O - P₁, V = P₂-P₁
// 3) sV = U + rD                        cross product with D (distributive)
// 4) s(V × D) = U × D  + r(D × D)       D × D = 0
// 5) s(V × D) = U × D                   let V × D = Z and U × D = X
// 6) sZ = X                             dot product 𝘇 to each side
// 7) sZ ⋅ Z = X ⋅ Z                     divide by Z ⋅ Z
// 8) s = (X ⋅ Z)/(Z ⋅ Z)
// If s ∉ [0, 1] the intersections is invalid. If s ∈ [0, 1],
// 1) O + rD = P₁ + sV                   subtracting O from both sides
// 2) rD = -U + sV                       cross product with 𝘃
// 3) r(D × V) = -U × V + s(V × V)       V × V = 0
// 4) r(D × V) = -U × V                  using D × V = -(V × D)
// 5) r(V × D) = U × V                   let U × V = Y
// 6) rZ = Y                             dot product Z to each side
// 7) r(Z ⋅ Z) = Y ⋅ Z                   divide by (Z ⋅ Z)
// 9) r = (Y ⋅ Z)/(Z ⋅ Z)
//
// The cross product of two vectors in the plane is a vector of the form (0, 0, k),
// hence, in 2D:
// s = (X ⋅ Z)/(Z ⋅ Z) = x₃/z₃
// r = (Y ⋅ Z)/(Z ⋅ Z) = y₃/z₃
// This result is valid if s ∈ [0, 1]
PURE HOSTDEV constexpr auto
intersect(Ray2 const & ray, LineSegment2 const & line) noexcept -> F
{
  Vec2<F> const v(line[1][0] - line[0][0], line[1][1] - line[0][1]);
  Vec2<F> const u(ray.origin()[0] - line[0][0], ray.origin()[1] - line[0][1]);

  F const z = v.cross(ray.direction());

  F const s = u.cross(ray.direction()) / z;
  F r = u.cross(v) / z;

  if (s < 0 || 1 < s) {
    r = inf_distance;
  }
  return r;
}

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

PURE HOSTDEV constexpr auto
intersect(Ray2 const & ray, QuadraticSegment2 const & q) noexcept -> Vec2<F>
{
  // NOLINTBEGIN(readability-identifier-naming) mathematical notation
  // This code is called very frequently so we sacrifice readability for speed.
  // Mainly, we want to ensure temporaries are not created.
  Vec2<F> const v01(q[1][0] - q[0][0], q[1][1] - q[0][1]); // q[1] - q[0]
  Vec2<F> const v02(q[2][0] - q[0][0], q[2][1] - q[0][1]); // q[2] - q[0]
  Vec2<F> const v12(q[2][0] - q[1][0], q[2][1] - q[1][1]); // q[2] - q[1]

  Vec2<F> const A(-2 * (v02[0] + v12[0]), -2 * (v02[1] + v12[1])); // -2(V₁₃ + V₂₃)
  Vec2<F> const B(3 * v02[0] + v12[0], 3 * v02[1] + v12[1]);       // 3V₁₃ + V₂₃
  // Vec2<F> const C = q[0];

  // Vec2<F> const D = ray.d;
  // Vec2<F> const O = ray.o;

  Vec2<F> const voc(q[0][0] - ray.origin()[0], q[0][1] - ray.origin()[1]); // C - O

  F const a = A.cross(ray.direction());   // (A × D)ₖ
  F const b = B.cross(ray.direction());   // (B × D)ₖ
  F const c = voc.cross(ray.direction()); // ((C - O) × D)ₖ

  Vec2<F> result(inf_distance, inf_distance);
#if UM2_ENABLE_FLOAT64
  F constexpr epsilon = 1e-8;
#else
  F constexpr epsilon = 1e-8f;
#endif

  if (um2::abs(a) < epsilon) { 
    F const s = -c / b;
    if (0 <= s && s <= 1) {
      Point2 const P(s * (s * A[0] + B[0]) + voc[0], s * (s * A[1] + B[1]) + voc[1]);
      result[0] = dot(P, ray.direction()) / ray.direction().squaredNorm();
    }
    return result;
  }
  F const disc = b * b - 4 * a * c;
  if (disc < 0) {
    return result;
  }

  F const s1 = (-b - um2::sqrt(disc)) / (2 * a);
  F const s2 = (-b + um2::sqrt(disc)) / (2 * a);
  if (0 <= s1 && s1 <= 1) {
    Point2 const P(s1 * (s1 * A[0] + B[0]) + voc[0], s1 * (s1 * A[1] + B[1]) + voc[1]);
    result[0] = dot(P, ray.direction()) / ray.direction().squaredNorm();
  }
  if (0 <= s2 && s2 <= 1) {
    Point2 const P(s2 * (s2 * A[0] + B[0]) + voc[0], s2 * (s2 * A[1] + B[1]) + voc[1]);
    result[1] = dot(P, ray.direction()) / ray.direction().squaredNorm();
  }
  // NOLINTEND(readability-identifier-naming)
  return result;
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::boundingBox() const noexcept -> AxisAlignedBox<D>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// distanceTo
//==============================================================================

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::squaredDistanceTo(Vertex const & p) const noexcept -> F
{
  F const r = pointClosestTo(p);
  Vertex const p_closest = (*this)(r);
  return p_closest.squaredDistanceTo(p);
}

template <Size P, Size N, Size D>
PURE HOSTDEV constexpr auto
Dion<P, N, D>::distanceTo(Vertex const & p) const noexcept -> F
{
  return um2::sqrt(squaredDistanceTo(p));
}

} // namespace um2
