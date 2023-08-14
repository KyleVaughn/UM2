namespace um2
{

// -------------------------------------------------------------------
// Triangle
// -------------------------------------------------------------------

template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Triangle<D, T> const & tri, R const r, S const s) noexcept -> Point<D, T>
{
  // (1 - r - s) v0 + r v1 + s v2
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = 1 - rr - ss;
  // T const w1 = rr;
  // T const w2 = ss;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * tri[0][i] + rr * tri[1][i] + ss * tri[2][i];
  }
  return result;
}

// -------------------------------------------------------------------
// Quadrilateral
// -------------------------------------------------------------------

template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(Quadrilateral<D, T> const & quad, R const r, S const s) noexcept
    -> Point<D, T>
{
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = (1 - rr) * (1 - ss);
  T const w1 = rr * (1 - ss);
  T const w2 = rr * ss;
  T const w3 = (1 - rr) * ss;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * quad[0][i] + w1 * quad[1][i] + w2 * quad[2][i] + w3 * quad[3][i];
  }
  return result;
}

// -------------------------------------------------------------------
// QuadraticTriangle
// -------------------------------------------------------------------
template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticTriangle<D, T> const & tri6, R const r, S const s) noexcept
    -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  // Factoring out the common terms
  T const tt = 1 - rr - ss;
  T const w0 = tt * (2 * tt - 1);
  T const w1 = rr * (2 * rr - 1);
  T const w2 = ss * (2 * ss - 1);
  T const w3 = 4 * rr * tt;
  T const w4 = 4 * rr * ss;
  T const w5 = 4 * ss * tt;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * tri6[0][i] + w1 * tri6[1][i] + w2 * tri6[2][i] + w3 * tri6[3][i] +
                w4 * tri6[4][i] + w5 * tri6[5][i];
  }
  return result;
}

// -------------------------------------------------------------------
// QuadraticQuadrilateral
// -------------------------------------------------------------------
template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
interpolate(QuadraticQuadrilateral<D, T> const & quad8, R const r, S const s) noexcept
    -> Point<D, T>
{
  T const xi = 2 * static_cast<T>(r) - 1;
  T const eta = 2 * static_cast<T>(s) - 1;
  T const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,
                  (1 + xi) * (1 - eta) * (xi - eta - 1) / 4,
                  (1 + xi) * (1 + eta) * (xi + eta - 1) / 4,
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,
                  (1 - xi * xi) * (1 - eta) / 2,
                  (1 - eta * eta) * (1 + xi) / 2,
                  (1 - xi * xi) * (1 + eta) / 2,
                  (1 - eta * eta) * (1 - xi) / 2};
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w[0] * quad8[0][i] + w[1] * quad8[1][i] + w[2] * quad8[2][i] +
                w[3] * quad8[3][i] + w[4] * quad8[4][i] + w[5] * quad8[5][i] +
                w[6] * quad8[6][i] + w[7] * quad8[7][i];
  }
  return result;
}

} // namespace um2
