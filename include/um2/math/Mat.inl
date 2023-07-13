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

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

// From a list of columns
template <Size M, Size N, typename T>
template <std::same_as<Vec<M, T>>... Cols>
requires(sizeof...(Cols) == N) HOSTDEV
    constexpr Mat<M, N, T>::Mat(Cols... in_cols) noexcept
    : cols{in_cols...}
{
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

template <typename T>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming)
operator*(Mat2x2<T> const & A, Vec2<T> const & x) noexcept -> Vec2<T>
{
  return Vec2<T>{A(0, 0) * x[0] + A(0, 1) * x[1], A(1, 0) * x[0] + A(1, 1) * x[1]};
}

template <typename T>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming)
operator*(Mat3x3<T> const & A, Vec3<T> const & x) noexcept -> Vec3<T>
{
  return Vec3<T>{A(0, 0) * x[0] + A(0, 1) * x[1] + A(0, 2) * x[2],
                 A(1, 0) * x[0] + A(1, 1) * x[1] + A(1, 2) * x[2],
                 A(2, 0) * x[0] + A(2, 1) * x[1] + A(2, 2) * x[2]};
}

} // namespace um2
