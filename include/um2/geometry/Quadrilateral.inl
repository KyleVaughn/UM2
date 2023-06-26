namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto Quadrilateral<D, T>::operator[](Size i)
    -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto Quadrilateral<D, T>::operator[](Size i) const
    -> Point<D, T> const &
{
  return vertices[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  return ((1 - rr) * (1 - ss)) * vertices[0] + 
               (rr * (1 - ss)) * vertices[1] + 
                     (rr * ss) * vertices[2] +
               ((1 - rr) * ss) * vertices[3];
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto Quadrilateral<D, T>::jacobian(R r, S s) const noexcept
    -> Mat<D, 2, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  Mat<D, 2, T> jac;
  jac.col(0) = (1 - ss) * (vertices[1] - vertices[0]) - ss * (vertices[3] - vertices[2]);
  jac.col(1) = (1 - rr) * (vertices[3] - vertices[0]) - rr * (vertices[1] - vertices[2]);
  return jac;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto Quadrilateral<D, T>::edge(Size i) const noexcept
    -> LineSegment<D, T>
{
  return (i == 3) ? LineSegment<D, T>(vertices[3], vertices[0])    
                  : LineSegment<D, T>(vertices[i], vertices[i + 1]);    
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept
    -> bool requires(D == 2)
{
  return areCCW(vertices[0], vertices[1], p) &&
         areCCW(vertices[1], vertices[2], p) &&
         areCCW(vertices[2], vertices[3], p) &&
         areCCW(vertices[3], vertices[0], p);
}

// -------------------------------------------------------------------
// See polytope.inl for
// -------------------------------------------------------------------
// area
// centroid
// boundingBox

} // namespace um2
