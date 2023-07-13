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
  // If an add or sub is 4 cycles and mul is 7, then we can solve for the weights quickly
  // using the following:
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
// isLeft
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2, "isLeft is only defined for 2D");
  // If the point is in the bounding box of the segment,
  // we need to check if the point is left of the segment.
  // To do this we must find the point on Q that is closest to P.
  // At this Q(r) we compute Q'(r) × (P - Q(r)). If this quantity is
  // positive, then P is left of the segment.
  //
  // To compute Q_nearest, we find r which minimizes ‖P - Q(r)‖.
  // This r also minimizes ‖P - Q(r)‖².
  // It can be shown that this is equivalent to finding the minimum of the
  // quartic function
  // ‖P - Q(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
  // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
  // W = P - P₁
  // a = 2(A ⋅ A)
  // b = 3(A ⋅ B)
  // c = [(B  ⋅ B) - 2(A ⋅W)]
  // d = -(B ⋅ W)
  // Lagrange's method is used to find the roots.
  // (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)

  //  Q(r) = C + rB + r²A,
  // where
  //  C = P₁
  //  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  //  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // and
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  // Q′(r) = B + 2rA,

  return areCCW(v[0], v[1], p);
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
  // If a = 0, then the segment is a line.
  if (a < static_cast<T>(1e-6)) {
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
  // Compare the extrema with the segment's endpoints to find the AABox
  Vec<D, T> v02 = v[2] - v[0];
  Vec<D, T> v12 = v[2] - v[1];

  Vec<D, T> b;
  Vec<D, T> a;
  for (Size i = 0; i < D; ++i) {
    b[i] = 3 * v02[i] + v12[i];
    a[i] = -2 * (v02[i] + v12[i]);
  }

  Vec<D, T> r;
  for (Size i = 0; i < D; ++i) {
    // r_i = -B_i / (2A_i)
    r[i] = -b[i] / (2 * a[i]);
  }

  Point<D, T> minima = v[0];
  minima.min(v[1]);
  Point<D, T> maxima = v[0];
  maxima.max(v[1]);
  for (Size i = 0; i < D; ++i) {
    // Not redundant for non-trivial segments
    // NOLINTNEXTLINE(misc-redundant-expression)
    if (0 < r[i] && r[i] < 1) {
      T const stationary = v[0][i] + r[i] * (b[i] + r[i] * a[i]);
      minima[i] = um2::min(minima[i], stationary);
      maxima[i] = um2::max(maxima[i], stationary);
    }
  }

  return AxisAlignedBox<D, T>{minima, maxima};
}

} // namespace um2
