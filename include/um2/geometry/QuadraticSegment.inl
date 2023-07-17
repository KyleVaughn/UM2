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
  static_assert(D == 2, "isLeft is only defined for 2D");

  // We want to rotate the segment so that v[0] is at the origin and v[1] is on the
  // x-axis. We then take note of the sign of the rotated v[2] to determine if the
  // segment curves left or right. We will orient the segment so that it curves right.
  //     Compute the rotation matrix
  Vec2<T> const v01 = v[1] - v[0];
  T const v01_norm = v01.norm();
  //     We can avoid a matrix multiplication by using the fact that the y-coordinate of
  //     v1_r is zero.
  Point2<T> const v1_r(v01_norm, static_cast<T>(0));
  Vec2<T> const v01_normalized = v01 / v01_norm;
  //     NOLINTBEGIN(readability-identifier-naming)
  Mat2x2<T> const R(um2::Vec<D, T>(v01_normalized[0], -v01_normalized[1]),
                    um2::Vec<D, T>(v01_normalized[1], v01_normalized[0]));
  Vec2<T> const v02 = v[2] - v[0];
  Point2<T> v2_r = R * v02;
  Point2<T> p_r = R * (p - v[0]);
  bool const curves_right = v2_r[1] >= 0;
  if (!curves_right) {
    // Flip the y-coordinates to be greater than or equal to zero
    v2_r[1] = -v2_r[1];
    p_r[1] = -p_r[1];
  }
  // If v2_r[1] < epsilonDistance, then the segment is straight and we can use the cross
  // product test to return early.
  bool const is_straight = v2_r[1] <= epsilonDistance<T>();
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
  // bounding box, then we can treat the segment like a straight line.
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
  T const Bx = 4 * v2_r[0] - v1_r[0];
  T const By = 4 * v2_r[1]; // Positive
  T const Ax = -Bx + v1_r[0];
  T const Ay = -By; // Negative
  // Handle the case where the point is above the bounding box
  {
    T ymax = v2_r[1];
    // Determine ymax
    // r_i = -B_i / (2A_i)
    T const half_b = By / 2;   // Positive
    T const ry = -half_b / Ay; // Positive
    // Only consider the stationary point if it is in the interval [0, 1]
    // Since ry is strictly positive, we only need to check if ry < 1
    if (ry < 1) {
      // x_i = Q(r_i) = - B² / (4A) = r(B/2)
      ymax = ry * half_b;
    }
    // Check if the point is above the BB
    if (ymax <= p_r[1]) {
      return curves_right;
    }
  }
  // Handle the case where the point is to the left or the right of the bounding box
  {
    T xmin = 0;
    T xmax = v1_r[0];
    // A is effectively 4 * (midpoint of v01 - v2_r), hence if Ax is small, then the
    // segment is effectively straight and we known xmin = 0 and xmax = v1_r[0]
    if (um2::abs(Ax) > 4 * epsilonDistance<T>()) {
      // r_i = -B_i / (2A_i)
      T const half_b = Bx / 2;
      T const rx = -half_b / Ax;
      // NOLINTNEXTLINE(misc-redundant-expression)
      if (0 < rx && rx < 1) {
        // x_i = Q(r_i) = - B² / (4A) = r(B/2)
        T const x_stationary = rx * half_b;
        xmin = um2::min(xmin, x_stationary);
        xmax = um2::max(xmax, x_stationary);
      }
    }
    if (p_r[0] <= xmin || xmax <= p_r[0]) {
      // Since the point is in the y-range of the AABB, the point will be
      // left of the segment if the segment curves right
      // if the segment curves left.
      return curves_right;
    }
  }
  // If the point is in the bounding box of the segment,
  // we will find the point on the segment that shares the same x-coordinate
  //  Q(r) = C + rB + r²A = P
  // Hence we wish to solve 0 = -P_x + rB_x + r²A_x for r.
  // This is a quadratic equation, which has two potential solutions.
  // r = (-b ± √(b² - 4ac)) / 2a
  // if A[0] == 0, then we have a well-behaved quadratic segment, and there is one root.
  // This is the expected case.
  if (um2::abs(Ax) < 4 * epsilonDistance<T>()) [[likely]] {
    // This is a linear equation, so there is only one root.
    T const r = p_r[0] / Bx; // B[0] != 0, otherwise the segment would be degenerate.
    // We know the point is in the AABB of the segment, so we expect r to be in [0, 1]
    assert(0 <= r && r <= 1);
    T const Q_y = r * (By + r * Ay);
    // if p_r < Q_y, then the point is to the right of the segment
    return (p_r[1] <= Q_y) ? !curves_right : curves_right;
  } else {
    // Two roots.
    T const disc = Bx * Bx + 4 * Ax * p_r[0];
    assert(disc >= 0);
    T const r1 = (-Bx + um2::sqrt(disc)) / (2 * Ax);
    T const r2 = (-Bx - um2::sqrt(disc)) / (2 * Ax);
    T const Q_y1 = r1 * (By + r1 * Ay);
    T const Q_y2 = r2 * (By + r2 * Ay);
    T const Q_ymin = um2::min(Q_y1, Q_y2);
    T const Q_ymax = um2::max(Q_y1, Q_y2);
    bool const contained_in_curve = Q_ymin <= p_r[1] && p_r[1] <= Q_ymax;
    return contained_in_curve ? !curves_right : curves_right;
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

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::enclosedArea() const noexcept -> T
{
  static_assert(D == 2, "enclosedArea() is only defined for 2D segments");
  // The area bounded by the segment and the line between the endpoints is
  // 4/3 of the area of the triangle formed by the vertices.
  // Assumes that the segment is convex.
  T const two_thirds = static_cast<T>(2) / static_cast<T>(3);
  Vec<D, T> const v02 = v[2] - v[0];
  Vec<D, T> const v01 = v[1] - v[0];
  return two_thirds * v02.cross(v01);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::enclosedCentroid() const noexcept -> Point<D, T>
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
  Vec2<T> const v12 = v[1] - v[0];
  Vec2<T> const four_v13 = 4 * (v[2] - v[0]);
  Vec2<T> const u1 = v12.normalized();
  Vec2<T> const u2(-u1[1], u1[0]);
  // NOLINTBEGIN(readability-identifier-naming)
  Mat2x2<T> const U(u1, u2);
  Vec2<T> const Cu(u1.dot((3 * v12 + four_v13)) / 10, u2.dot(four_v13) / 10);
  return U * Cu + v[0];
  // NOLINTEND(readability-identifier-naming)
}

} // namespace um2
