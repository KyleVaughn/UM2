namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Triangle<D, T>::operator[](len_t i)
    -> Point<D, T> &
{
  return this->vertices[i];
}

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Triangle<D, T>::operator[](len_t i) const
    -> Point<D, T> const &
{
  return this->vertices[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <len_t D, typename T>
template <typename R, typename S>
UM2_PURE UM2_HOSTDEV constexpr auto Triangle<D, T>::operator()(R const r,
                                                               S const s) const noexcept
    -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w[3] = {1 - rr - ss, rr, ss};
  return w[0] * this->vertices[0] + w[1] * this->vertices[1] + w[2] * this->vertices[2];
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <len_t D, typename T>
template <typename R, typename S>
UM2_PURE UM2_HOSTDEV constexpr auto Triangle<D, T>::jacobian(R /*r*/,
                                                             S /*s*/) const noexcept
    -> Mat<D, 2, T>
{
  Mat<D, 2, T> jac;
  jac.col(0) = this->vertices[1] - this->vertices[0];
  jac.col(1) = this->vertices[2] - this->vertices[0];
  return jac;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Triangle<D, T>::edge(len_t i) const noexcept
    -> LineSegment<D, T>
{
  if (i == 2) {
    return LineSegment<D, T>(this->vertices[2], this->vertices[0]);
  }
  return LineSegment<D, T>(this->vertices[i], this->vertices[i + 1]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::contains(Point<D, T> const & p) const noexcept -> bool requires(D == 2)
{
  return areCCW(this->vertices[0], this->vertices[1], p) &&
         areCCW(this->vertices[1], this->vertices[2], p) &&
         areCCW(this->vertices[2], this->vertices[0], p);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto area(Triangle<2, T> const & tri) noexcept -> T
{
  return cross2(tri[1] - tri[0], tri[2] - tri[0]) / 2;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto area(Triangle<3, T> const & tri) noexcept -> T
{
  return (tri[1] - tri[0]).cross(tri[2] - tri[0]).norm() / 2;
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto centroid(Triangle<D, T> const & tri) noexcept
    -> Point<D, T>
{
  return (tri[0] + tri[1] + tri[2]) / 3;
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto boundingBox(Triangle<D, T> const & tri) noexcept
    -> AABox<D, T>
{
  return boundingBox(tri.vertices);
}

} // namespace um2
