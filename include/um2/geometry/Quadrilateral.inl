namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr Quadrilateral<D, T>::Polytope(Point<D, T> const & p0,
                                                Point<D, T> const & p1,
                                                Point<D, T> const & p2,
                                                Point<D, T> const & p3) noexcept
    : v{p0, p1, p2, p3}
{
}

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  return interpolate(*this, r, s);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// edge
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::getEdge(Size i) const noexcept -> LineSegment<D, T>
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// contains
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  return um2::contains(*this, p);
}

//==============================================================================
// area
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::area() const noexcept -> T
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
{
  return um2::centroid(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isConvex
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isConvex() const noexcept -> bool
{
  return um2::isConvex(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isCCW() const noexcept -> bool
{
  return um2::isCCW(*this);
}

} // namespace um2
