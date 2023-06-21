namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::operator[](len_t i) -> Point<D, T> &
{
  return this->vertices[i];
}

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::operator[](len_t i) const -> Point<D, T> const &
{
  return this->vertices[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <len_t D, typename T>
template <typename R, typename S>
UM2_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
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
UM2_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::jacobian(R /*r*/, S /*s*/) const noexcept -> Mat<D, 2, T>
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
UM2_PURE UM2_HOSTDEV constexpr auto
Triangle<D, T>::edge(len_t i) const noexcept -> LineSegment<D, T>
{
  return (i == 2) ? LineSegment<D, T>(this->vertices[2], this->vertices[0])
                  : LineSegment<D, T>(this->vertices[i], this->vertices[i + 1]);
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
// See polytope.inl for
// -------------------------------------------------------------------
// area
// centroid
// boundingBox

} // namespace um2
