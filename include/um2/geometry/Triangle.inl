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
  T const w0 = (1 - rr - ss);
  // T const w1 = rr;
  // T const w2 = ss;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * vertices[0][i] + rr * vertices[1][i] + ss * vertices[2][i];
  }
  return result;
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
  for (Size i = 0; i < D; ++i) {
    jac(i, 0) = vertices[1][i] - vertices[0][i];
    jac(i, 1) = vertices[2][i] - vertices[0][i];
  }
  return jac;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::edge(Size i) const noexcept -> LineSegment<D, T>
{
  assert(i < 3);
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
  return areCCW(vertices[0], vertices[1], p) && 
         areCCW(vertices[1], vertices[2], p) &&
         areCCW(vertices[2], vertices[0], p);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::area() const noexcept -> T
{
  Vec<D, T> ab;
  Vec<D, T> ac;
  for (Size i = 0; i < D; ++i) {
    ab[i] = vertices[1][i] - vertices[0][i];
    ac[i] = vertices[2][i] - vertices[0][i];
  }
  if constexpr (D == 2) {
    return ab.cross(ac) / 2;
  } else if constexpr (D == 3) {
    return ab.cross(ac).norm() / 2;
  } else {
    static_assert(D == 2 || D == 3,
                  "Triangle::area() is only defined for 2D and 3D triangles");
  }
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::centroid() const noexcept -> Point<D, T>
{
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = (vertices[0][i] + vertices[1][i] + vertices[2][i]);
  }
  return result /= 3;
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
