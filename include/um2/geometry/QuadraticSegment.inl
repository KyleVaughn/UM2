#include <iostream>

namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr QuadraticSegment<D, T>::Polytope(Point<D, T> const & p0,
                                                   Point<D, T> const & p1,
                                                   Point<D, T> const & p2) noexcept
{
  v[0] = p0;
  v[1] = p1;
  v[2] = p2;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  T const rr = static_cast<T>(r);
  // We factor out the common terms to reduce the number of multiplications
  T const two_rr = 2 * rr;
  T const rr_1 = rr - 1;
  T const x = two_rr * rr_1;

  T const w0 = x - rr_1;
  T const w1 = (two_rr - 1) * rr;
  T const w2 = -2 * x;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * v[0][i] + w1 * v[1][i] + w2 * v[2][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::jacobian(R r) const noexcept -> Vec<D, T>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  T const w0 = 4 * static_cast<T>(r) - 3;
  T const w1 = 4 * static_cast<T>(r) - 1;
  Vec<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * (v[0][i] - v[2][i]) + w1 * (v[1][i] - v[2][i]);
  }
  return result;
}

// -------------------------------------------------------------------
// getRotation
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::getRotation() const noexcept -> Mat<D, D, T>
{
  return LineSegment<D, T>(v[0], v[1]).getRotation();
}

// -------------------------------------------------------------------
// isStraight
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isStraight() const noexcept -> bool
{
  // A slightly more optimized version of doing:
  // LineSegment(v[0], v[1]).distanceTo(v[2]) < epsilonDistance
  //
  // Compute the point on the line v[0] + r * (v[1] - v[0]) that is closest to v[2]
  Vec<D, T> const v01 = v[1] - v[0];
  T const r = (v[2] - v[0]).dot(v01) / v01.squaredNorm();
  // If r is outside the range [0, 1], then the segment is not straight
  if (r < 0 || r > 1) {
    return false;
  }
  // Compute the point on the line
  Vec<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = v[0][i] + r * v01[i];
  }
  // Check if the point is within epsilon distance of v[2]
  return isApprox(p, v[2]);
}

// -------------------------------------------------------------------
// curvesLeft
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::curvesLeft() const noexcept -> bool
{
  static_assert(D == 2, "curvesLeft is only defined for 2D");
  // If the segment is not straight, then we can compute the cross product of the
  // vectors from v[0] to v[1] and v[0] to v[2]. If the cross product is positive,
  // then the segment curves left. If the cross product is negative, then the segment
  // curves right.
  Vec<D, T> const v01 = v[1] - v[0];
  Vec<D, T> const v02 = v[2] - v[0];
  return v01.cross(v02) >= 0;
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  // This routine has previously been a major bottleneck, so some readability
  // has been sacrificed for performance.
  // Mainly, we could use some member functions to make this more readable, but 
  // this results in redundant computations. Instead, we will do the computations
  // once and then use them multiple times.
  static_assert(D == 2, "isLeft is only defined for 2D");
  // Algorithm:
  // 1) Translate by -v[0] and rotate the segment so that v[0] is at the origin and
  //  v[1] is on the x-axis.
  // 2) Compute the AABB of the segment.
  // 3) If p_rotated is outside the AABB of the segment, we may return early. 
  // 4) If p_rotated is inside the AABB of the segment, then we find the point on the
  //    segment that shares the same x-coordinate as p_rotated.
  // 5) If p_rotated is above this point, then the segment curves left. Otherwise, it
  //   curves right.
 
  // Compute rotation matrix and rotated points 
  Vec2<T> const v01 = v[1] - v[0];
  Vec2<T> const v02 = v[2] - v[0];
  Vec2<T> const v01_normalized = v01.normalized();
  // NOLINTBEGIN(readability-identifier-naming)
  Mat2x2<T> const R(
      um2::Vec<D, T>(v01_normalized[0], -v01_normalized[1]), 
      um2::Vec<D, T>(v01_normalized[1],  v01_normalized[0]));
  Point2<T> const p_rotated = R * (p - v[0]);
  Point2<T> const v1_rotated = R * v01; 
  Point2<T> const v2_rotated = R * v02;

  // Compute the bounding box
  //  Q(r) = C + rB + r²A 
  // where
  //  C = (0, 0)
  //  B =  4 * v2_rotated - v1_rotated
  //  A = -4 * v2_rotated + 2 * v1_rotated = -B + v1_rotated
  // Q′(r) = B + 2rA,
  // (r_i,...) = -B / (2A)
  Vec2<T> const B(4 * v2_rotated[0] - v1_rotated[0], 
                  4 * v2_rotated[1] - v1_rotated[1]);
  Vec2<T> const A(-B[0] + v1_rotated[0], 
                  -B[1] + v1_rotated[1]);
  bool const curves_left = v2_rotated[1] <= 0;
  T ymax = curves_left ? 0 : v2_rotated[1];
  T ymin = curves_left ? v2_rotated[1] : 0;
  if (um2::abs(A[1]) > 4 * epsilonDistance<T>()) {
    // r_i = -B_i / (2A_i)
    T const half_by = B[1] / 2;
    T const ry = -half_by / A[1];
    // NOLINTNEXTLINE(misc-redundant-expression)
    if (0 < ry && ry < 1) {
      // x_i = Q(r_i) = - B² / (4A) = r(B/2)
      T const y_stationary = ry * half_by;
      ymin = um2::min(ymin, y_stationary);
      ymax = um2::max(ymax, y_stationary);
    }
  }
  // Check if the point is above or below the AABB 
  if (ymax <= p_rotated[1]) {
    return true;
  }
  if (p_rotated[1] <= ymin) {
    return false;
  }
  T xmin = 0;
  T xmax = v1_rotated[0];
  if (um2::abs(A[0]) > 4 * epsilonDistance<T>()) {
    // r_i = -B_i / (2A_i)
    T const half_bx = B[0] / 2;
    T const rx = -half_bx / A[0];
    // NOLINTNEXTLINE(misc-redundant-expression)
    if (0 < rx && rx < 1) {
      // x_i = Q(r_i) = - B² / (4A) = r(B/2)
      T const x_stationary = rx * half_bx;
      xmin = um2::min(xmin, x_stationary);
      xmax = um2::max(xmax, x_stationary);
    }
  }
  if (p_rotated[0] <= xmin || xmax <= p_rotated[0]) {
    // Since the point is in the y-range of the AABB, the point will be
    // left of the segment if the segment curves right and right of the segment
    // if the segment curves left.
    return !curves_left; 
  }
  // If the point is in the bounding box of the segment,
  // we will find the point on the segment that shares the same x-coordinate
  //  Q(r) = C + rB + r²A = P
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A[0] == 0, then we have a well-behaved quadratic segment, and there is one root.
  // This is the expected case.
  if (um2::abs(A[0]) < 4 * epsilonDistance<T>()) [[likely]] {
    // This is a linear equation, so there is only one root.
    T const r = p_rotated[0] / B[0]; // B[0] != 0, otherwise the segment would be degenerate.
    assert(0 <= r && r <= 1);
    T const Q_y = r * (B[1] + r * A[1]);
    return Q_y <= p_rotated[1]; 
  } else {
    // Two roots.
    T const disc = B[0] * B[0] + 4 * A[0] * p_rotated[0];
    assert(disc >= 0);
    T const r1 = (-B[0] + um2::sqrt(disc)) / (2 * A[0]);
    T const r2 = (-B[0] - um2::sqrt(disc)) / (2 * A[0]);
    T const Q_y1 = r1 * (B[1] + r1 * A[1]);
    T const Q_y2 = r2 * (B[1] + r2 * A[1]);
    T const Q_ymin = um2::min(Q_y1, Q_y2);
    T const Q_ymax = um2::max(Q_y1, Q_y2);
    if (Q_ymax <= p_rotated[1]) {
      return true; 
    }
    if (p_rotated[1] <= Q_ymin) {
      return false; 
    }
    return curves_left;
  }
  // NOLINTEND(readability-identifier-naming)
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::length() const noexcept -> T
{
  // Turn off variable naming convention warning for this function, since we will use
  // capital letters to denote vectors.
  // NOLINTBEGIN(readability-identifier-naming)

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
  Vec<D, T> const v13 = v[2] - v[0];
  Vec<D, T> const v23 = v[2] - v[1];
  Vec<D, T> A;
  for (Size i = 0; i < D; ++i) {
    A[i] = -2 * (v13[i] + v23[i]);
  }
  // Move computation of B to after exit.

  // ‖Q′(r)‖ =  √(4(A ⋅A)r² + 4(A ⋅B)r + B ⋅B) = √(ar² + br + c)
  // where
  // a = 4(A ⋅ A)
  // b = 4(A ⋅ B)
  // c = B ⋅ B

  T const a = 4 * A.squaredNorm();
  // 0 ≤ a, since a = 4(A ⋅ A)  = 4 ‖A‖², and 0 ≤ ‖A‖²
  // A = 4(midpoint of line - p3) -> a = 64 ‖midpoint of line - p3‖²
  // if a is small, then the segment is almost a straight line, and we can use the
  // distance between the endpoints as an approximation.
  if (a < 64 * epsilonDistanceSquared<T>()) { 
    return v[0].distanceTo(v[1]);
  }
  Vec<D, T> B;
  for (Size i = 0; i < D; ++i) {
    B[i] = 3 * v13[i] + v23[i];
  }
  T const b = 4 * A.dot(B);
  T const c = B.squaredNorm();

  // √(ar² + br + c) = √a √( (r + b₁)^2 + c₁)
  // where
  T const b1 = b / (2 * a);
  T const c1 = (c / a) - (b1 * b1);
  // The step above with division by a is safe, since a ≠ 0.

  // Let u = r + b₁, then
  // 1                       1 + b₁
  // ∫ √(ar² + br + c) dr = √a ∫ √(u² + c₁) du
  // 0                         b₁
  //
  // This is an integral that exists in common integral tables.
  // Evaluation of the resultant expression may be simplified by using

  T const lb = b1;
  T const ub = 1 + b1;
  T const L = um2::sqrt(c1 + lb * lb);
  T const U = um2::sqrt(c1 + ub * ub);
  T const atanh_u = um2::atanh(ub / U);
  T const atanh_l = um2::atanh(lb / L);

  return um2::sqrt(a) * (U + lb * (U - L) + c1 * (atanh_u - atanh_l)) / 2;
  // NOLINTEND(readability-identifier-naming)
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
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
  Vec<D, T> const v02 = v[2] - v[0];
  Vec<D, T> const v12 = v[2] - v[1];

  Point<D, T> minima = um2::min(v[0], v[1]);
  Point<D, T> maxima = um2::max(v[0], v[1]);
  for (Size i = 0; i < D; ++i) {
    T const a = -2 * (v02[i] + v12[i]);
    if (um2::abs(a) < 4 * epsilonDistance<T>()) {
      // The segment is almost a straight line, so the extrema are the endpoints.
      continue;
    }
    // r_i = -B_i / (2A_i)
    T const half_b = (3 * v02[i] + v12[i]) / 2;
    T const r = -half_b / a;
    // if r is not in [0, 1], then the extrema are not on the segment, hence
    // the segment's endpoints are the extrema.
    // NOLINTNEXTLINE(misc-redundant-expression)
    if (0 < r && r < 1) {
      // x_i = Q(r_i) = P₁ - B² / (4A) = P₁ + r(B/2)
      T const x = v[0][i] + r * half_b; 
      minima[i] = um2::min(minima[i], x); 
      maxima[i] = um2::max(maxima[i], x);
    }
  }
  return AxisAlignedBox<D, T>{minima, maxima};
}

} // namespace um2
