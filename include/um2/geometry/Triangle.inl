namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return vertices[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  return (1 - rr - ss) * vertices[0] + rr * vertices[1] + ss * vertices[2];
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Triangle<D, T>::jacobian(R /*r*/, S /*s*/) const noexcept -> Mat<D, 2, T>
{
  Mat<D, 2, T> jac;
  jac.col(0) = vertices[1] - vertices[0];
  jac.col(1) = vertices[2] - vertices[0];
  return jac;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::edge(Size i) const noexcept -> LineSegment<D, T>
{
  return (i == 2) ? LineSegment<D, T>(vertices[2], vertices[0])
                  : LineSegment<D, T>(vertices[i], vertices[i + 1]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::contains(Point<D, T> const & p) const noexcept -> bool requires(D == 2)
{
  return areCCW(vertices[0], vertices[1], p) && areCCW(vertices[1], vertices[2], p) &&
         areCCW(vertices[2], vertices[0], p);
}

// -------------------------------------------------------------------
// area 
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::area() const noexcept -> T
{
  if constexpr (D == 2) {
    return cross2(vertices[1] - vertices[0], vertices[2] - vertices[0]) / 2;
  } else if constexpr (D == 3) {
    return (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]).norm() / 2; 
  } else {
    static_assert(D == 2 || D == 3, "Triangle::area() is only defined for 2D and 3D triangles");
  }
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::centroid() const noexcept -> Point<D, T>
{
  return (vertices[0] + vertices[1] + vertices[2]) / 3;
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(vertices);
}

} // namespace um2
