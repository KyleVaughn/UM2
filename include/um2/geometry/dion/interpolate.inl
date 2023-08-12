namespace um2
{

// -------------------------------------------------------------------
// LineSegment
// -------------------------------------------------------------------

template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
interpolate(LineSegment<D, T> const & l, R const r) noexcept -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = l[0][i] + rr * (l[1][i] - l[0][i]);
  }
  return result;
}

// -------------------------------------------------------------------
// QuadraticSegment
// -------------------------------------------------------------------
template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
interpolate(QuadraticSegment<D, T> const & q, R const r) noexcept -> Point<D, T>
{
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  T const rr = static_cast<T>(r);
  T const two_rr_1 = 2 * rr - 1;
  T const rr_1 = rr - 1;

  T const w0 = two_rr_1 * rr_1;
  T const w1 = two_rr_1 * rr;
  T const w2 = -4 * rr * rr_1;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * q[0][i] + w1 * q[1][i] + w2 * q[2][i];
  }
  return result;
}

} // namespace um2
