namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Size i) noexcept -> typename Mat<M, N, T>::Col &
{
  assert(i < N);
  return cols[i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Size i) const noexcept -> typename Mat<M, N, T>::Col const &
{
  assert(i < N);
  return cols[i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Size i, Size j) noexcept -> T &
{
  assert(i < M && j < N);
  return cols[j][i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Size i, Size j) const noexcept -> T const &
{
  assert(i < M && j < N);
  return cols[j][i];
}

} // namespace um2
