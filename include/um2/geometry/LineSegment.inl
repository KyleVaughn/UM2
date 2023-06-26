namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Size i) -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator[](Size i) const -> Point<D, T> const &
{
  return vertices[i];
}

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr LineSegment<D, T>::Polytope(Point<D, T> const & p0,
                                              Point<D, T> const & p1) noexcept
{
  vertices[0] = p0;
  vertices[1] = p1;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  return vertices[0] + static_cast<T>(r) * (vertices[1] - vertices[0]);
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::jacobian(R /*r*/) const noexcept -> Vec<D, T>
{
  return vertices[1] - vertices[0];
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
LineSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool requires(D == 2)
{
  return areCCW(vertices[0], vertices[1], p);
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
length(LineSegment<D, T> const & line) noexcept -> T
{
  return distance(line[0], line[1]);
}

} // namespace um2