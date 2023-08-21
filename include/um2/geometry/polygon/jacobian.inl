namespace um2
{

//==============================================================================
// Triangle
//==============================================================================

template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Triangle<D, T> const & t, R /*r*/, S /*s*/) noexcept -> Mat<D, 2, T>
{
  return Mat<D, 2, T>(t[1] - t[0], t[2] - t[0]);
}

//==============================================================================
// Quadrilateral
//==============================================================================

template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(Quadrilateral<D, T> const & q, R const r, S const s) noexcept -> Mat<D, 2, T>
{
  // jac.col(0) = (v1 - v0) - s (v3 - v2)
  // jac.col(1) = (v3 - v0) - r (v1 - v2)
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = 1 - ss;
  // T const w1 = ss;
  T const w2 = 1 - rr;
  // T const w3 = rr;
  Mat<D, 2, T> jac;
  for (Size i = 0; i < D; ++i) {
    jac(i, 0) = w0 * (q[1][i] - q[0][i]) - ss * (q[3][i] - q[2][i]);
    jac(i, 1) = w2 * (q[3][i] - q[0][i]) - rr * (q[1][i] - q[2][i]);
  }
  return jac;
}

//==============================================================================
// QuadraticTriangle
//==============================================================================
template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticTriangle<D, T> const & t6, R const r, S const s) noexcept
    -> Mat<D, 2, T>
{
  T const rr = static_cast<T>(4 * r);
  T const ss = static_cast<T>(4 * s);
  T const tt = rr + ss - 3;
  Mat<D, 2, T> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = tt * (t6[0][i] - t6[3][i]) + (rr - 1) * (t6[1][i] - t6[3][i]) +
                       ss * (t6[4][i] - t6[5][i]);
    result.col(1)[i] = tt * (t6[0][i] - t6[5][i]) + (ss - 1) * (t6[2][i] - t6[5][i]) +
                       rr * (t6[4][i] - t6[3][i]);
  }
  return result;
}

//==============================================================================
// QuadraticQuadrilateral
//==============================================================================
template <Size D, typename T, typename R, typename S>
PURE HOSTDEV constexpr auto
jacobian(QuadraticQuadrilateral<D, T> const & q, R const r, S const s) noexcept
    -> Mat<D, 2, T>
{
  T const xi = 2 * static_cast<T>(r) - 1;
  T const eta = 2 * static_cast<T>(s) - 1;
  T const xi_eta = xi * eta;
  T const xi_xi = xi * xi;
  T const eta_eta = eta * eta;
  T const w0 = (eta - eta_eta) / 2;
  T const w1 = (eta + eta_eta) / 2;
  T const w2 = (xi - xi_eta);
  T const w3 = (xi + xi_eta);
  T const w4 = 1 - eta_eta;
  T const w5 = (xi - xi_xi) / 2;
  T const w6 = (xi + xi_xi) / 2;
  T const w7 = eta - xi_eta;
  T const w8 = eta + xi_eta;
  T const w9 = 1 - xi_xi;
  Mat<D, 2, T> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = w0 * (q[0][i] - q[1][i]) + w1 * (q[2][i] - q[3][i]) +
                       w2 * (q[0][i] + q[1][i] - 2 * q[4][i]) +
                       w3 * (q[2][i] + q[3][i] - 2 * q[6][i]) + w4 * (q[5][i] - q[7][i]);
    result.col(1)[i] = w5 * (q[0][i] - q[3][i]) + w6 * (q[2][i] - q[1][i]) +
                       w7 * (q[0][i] + q[3][i] - 2 * q[7][i]) +
                       w8 * (q[1][i] + q[2][i] - 2 * q[5][i]) + w9 * (q[6][i] - q[4][i]);
  }
  return result;
}

} // namespace um2
