namespace um2
{

//==============================================================================
// LineSegment
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
length(LineSegment<D, T> const & l) noexcept -> T
{
  return (l[0]).distanceTo(l[1]);
}

//==============================================================================
// QuadraticSegment
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
length(QuadraticSegment<D, T> const & q) noexcept -> T
{
  // Turn off variable naming convention warning for this function, since we will use
  // capital letters to denote vectors.
  // NOLINTBEGIN(readability-identifier-naming) justification: mathematical convention

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

  Vec<D, T> const v13 = q[2] - q[0];
  Vec<D, T> const v23 = q[2] - q[1];
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

  Vec<D, T> B;
  for (Size i = 0; i < D; ++i) {
    B[i] = 3 * v13[i] + v23[i];
  }
  T const a = 4 * squaredNorm(A);
  T const b = 4 * dot(A, B);
  T const c = squaredNorm(B);

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
  // Numerical issues may cause the bounds to be slightly outside the range [-1, 1].
  // If we go too far outside this range, error out as something has gone wrong.
  assert(static_cast<T>(-1.0001) <= (lb / L) && (lb / L) <= static_cast<T>(1.0001));
  assert(static_cast<T>(-1.0001) <= (ub / U) && (ub / U) <= static_cast<T>(1.0001));
  T const arg_l = um2::clamp(lb / L, static_cast<T>(-0.99999), static_cast<T>(0.99999));
  T const arg_u = um2::clamp(ub / U, static_cast<T>(-0.99999), static_cast<T>(0.99999));
  T const atanh_l = um2::atanh(arg_l);
  T const atanh_u = um2::atanh(arg_u);
  T const result = um2::sqrt(a) * (U + lb * (U - L) + c1 * (atanh_u - atanh_l)) / 2;
  assert(0 <= result && result <= infiniteDistance<T>());
  return result;
  // NOLINTEND(readability-identifier-naming)
}

} // namespace um2
