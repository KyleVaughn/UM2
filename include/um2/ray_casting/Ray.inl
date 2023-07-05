namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr Ray<D, T>::Ray(Point<D, T> const & origin,
                                 Vec<D, T> const & direction) noexcept
    : o(origin),
      d(direction)
{
  assert(um2::abs(direction.norm() - static_cast<T>(1)) < static_cast<T>(1e-5));
}

} // namespace um2
