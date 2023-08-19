namespace um2
{
//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr QuadraticQuadrilateral<D, T>::Polytope(
    Point<D, T> const & p0, Point<D, T> const & p1, Point<D, T> const & p2,
    Point<D, T> const & p3, Point<D, T> const & p4, Point<D, T> const & p5,
    Point<D, T> const & p6, Point<D, T> const & p7) noexcept
    : v{p0, p1, p2, p3, p4, p5, p6, p7}
{
}

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator()(R const r, S const s) const noexcept
    -> Point<D, T>
{
  return interpolate(*this, r, s);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// edge
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::getEdge(Size i) const noexcept -> QuadraticSegment<D, T>
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// contains
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  return um2::contains(*this, p);
}

//==============================================================================
// linearPolygon
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::linearPolygon() const noexcept -> Quadrilateral<D, T>
{
  return um2::linearPolygon(*this);
}

//==============================================================================
// area
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::area() const noexcept -> T
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
{
  return um2::centroid(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::isCCW() const noexcept -> bool
{
  return um2::isCCW(*this);
}

} // namespace um2
