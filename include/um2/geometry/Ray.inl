namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr Ray<D, T>::Ray(Point<D, T> const & origin,
                                 Vec<D, T> const & direction) noexcept
    : o(origin),
      d(direction)
{
  assert(um2::abs(direction.squaredNorm() - static_cast<T>(1)) < static_cast<T>(1e-5));
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr auto 
Ray<D, T>::operator()(T const r) const noexcept -> Point<D, T> 
{
  Point<D, T> res;
  for (Size i = 0; i < D; ++i) { 
    res[i] = o[i] + r * d[i];
  }
  return res; 
}

} // namespace um2
