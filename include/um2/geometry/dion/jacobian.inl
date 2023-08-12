namespace um2
{

// -------------------------------------------------------------------
// LineSegment
// -------------------------------------------------------------------

template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
jacobian(LineSegment<D, T> const & l, R const /*r*/) noexcept -> Point<D, T>
{
  return l[1] - l[0];
}

// -------------------------------------------------------------------
// QuadraticSegment
// -------------------------------------------------------------------
template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
jacobian(QuadraticSegment<D, T> const & q, R const r) noexcept -> Point<D, T>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  T const w0 = 4 * static_cast<T>(r) - 3;
  T const w1 = 4 * static_cast<T>(r) - 1;
  Vec<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * (q[0][i] - q[2][i]) + w1 * (q[1][i] - q[2][i]);
  }
  return result;
}

} // namespace um2
