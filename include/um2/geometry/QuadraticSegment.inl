namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return vertices[i];
}

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr QuadraticSegment<D, T>::Polytope(Point<D, T> const & p0,
                                                   Point<D, T> const & p1,
                                                   Point<D, T> const & p2) noexcept
{
  vertices[0] = p0;
  vertices[1] = p1;
  vertices[2] = p2;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  // (2 * r - 1) * (r - 1) * v0 +    
  // (2 * r - 1) *  r      * v1 +    
  // -4 * r      * (r - 1) * v2
  T const rr = static_cast<T>(r);
  T const w0 = (2 * rr - 1) * (rr - 1);
  T const w1 = (2 * rr - 1) *  rr;
  T const w2 = -4 * rr      * (rr - 1);
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * this->vertices[0][i] +    
                w1 * this->vertices[1][i] +    
                w2 * this->vertices[2][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::jacobian(R r) const noexcept -> Vec<D, T>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  Vec<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = (4 * static_cast<T>(r) - 3) * (this->vertices[0][i] - this->vertices[2][i]) +
                (4 * static_cast<T>(r) - 1) * (this->vertices[1][i] - this->vertices[2][i]);
  }
  return result;
}

//// -------------------------------------------------------------------
//// isLeft
//// -------------------------------------------------------------------
//
//template <Size D, typename T>
//PURE HOSTDEV constexpr auto
//QuadraticSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool requires(D == 2)
//{
//  return areCCW(vertices[0], vertices[1], p);
//}
//
//// -------------------------------------------------------------------
//// length
//// -------------------------------------------------------------------
//
//template <Size D, typename T>
//PURE HOSTDEV constexpr auto
//QuadraticSegment<D, T>::length() const noexcept -> T
//{
//  return vertices[0].distanceTo(vertices[1]);
//}
//
//// -------------------------------------------------------------------
//// boundingBox
//// -------------------------------------------------------------------
//
//template <Size D, typename T>
//PURE HOSTDEV constexpr auto
//QuadraticSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
//{
//  return um2::boundingBox(vertices);
//}

} // namespace um2
