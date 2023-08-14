namespace um2
{

// -------------------------------------------------------------------
// LineSegment
// -------------------------------------------------------------------
// Defined in Polytope.hpp, since the bounding box of all linear
// polytopes is simply the bounding box of its vertices.

// -------------------------------------------------------------------
// QuadraticSegment
// -------------------------------------------------------------------
template <Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(QuadraticSegment<D, T> const & q) noexcept -> AxisAlignedBox<D, T>
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
  Vec<D, T> const v02 = q[2] - q[0];
  Vec<D, T> const v12 = q[2] - q[1];
  Point<D, T> minima = um2::min(q[0], q[1]);
  Point<D, T> maxima = um2::max(q[0], q[1]);
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
      T const x = q[0][i] + r * half_b;
      minima[i] = um2::min(minima[i], x);
      maxima[i] = um2::max(maxima[i], x);
    }
  }
  return AxisAlignedBox<D, T>{minima, maxima};
}

} // namespace um2
