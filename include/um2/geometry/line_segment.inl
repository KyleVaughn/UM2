namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto LineSegment<D, T>::operator[](len_t i)
    -> Point<D, T> &
{
  return this->vertices[i];
}

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto LineSegment<D, T>::operator[](len_t i) const
    -> Point<D, T> const &
{
  return this->vertices[i];
}

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_HOSTDEV constexpr LineSegment<D, T>::Polytope(Point<D, T> const & p0,
                                                  Point<D, T> const & p1) noexcept
{
  this->vertices[0] = p0;
  this->vertices[1] = p1;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <len_t D, typename T>
template <typename R>
UM2_PURE UM2_HOSTDEV constexpr auto
LineSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  return this->vertices[0] + static_cast<T>(r) * (this->vertices[1] - this->vertices[0]);
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <len_t D, typename T>
template <typename R>
UM2_PURE UM2_HOSTDEV constexpr auto LineSegment<D, T>::jacobian(R /*r*/) const noexcept
    -> Vec<D, T>
{
  return this->vertices[1] - this->vertices[0];
}

// -------------------------------------------------------------------
// isLeft
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto
LineSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool requires(D == 2)
{
  return areCCW(this->vertices[0], this->vertices[1], p);
}

// -------------------------------------------------------------------
// length
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto length(LineSegment<D, T> const & line) noexcept -> T
{
  return distance(line[0], line[1]);
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(LineSegment<D, T> const & line) noexcept
    -> AABox<D, T>
{
  return boundingBox(line.vertices);
}

} // namespace um2
