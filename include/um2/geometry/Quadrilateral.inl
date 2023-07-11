namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
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
  // (1 - r) (1 - s) v0 +
  // (    r) (1 - s) v1 +
  // (    r) (    s) v2 +
  // (1 - r) (    s) v3
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = (1 - rr) * (1 - ss);
  T const w1 = rr * (1 - ss);
  T const w2 = rr * ss;
  T const w3 = (1 - rr) * ss;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * vertices[0][i] + w1 * vertices[1][i] + w2 * vertices[2][i] +
                w3 * vertices[3][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  // jac.col(0) = (v1 - v0) - s (v3 - v2)
  // jac.col(1) = (v3 - v0) - r (v1 - v2)
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = 1 - ss;
  // T const w1 = ss;
  T const w2 = 1 - rr;
  // T const w3 = rr;
  Mat<D, 2, T> jac;
  for (Size i = 0; i < D; ++i) {
    jac(i, 0) =
        w0 * (vertices[1][i] - vertices[0][i]) - ss * (vertices[3][i] - vertices[2][i]);
    jac(i, 1) =
        w2 * (vertices[3][i] - vertices[0][i]) - rr * (vertices[1][i] - vertices[2][i]);
  }
  return jac;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::edge(Size i) const noexcept -> LineSegment<D, T>
{
  assert(i < 4);
  return (i == 3) ? LineSegment<D, T>(vertices[3], vertices[0])
                  : LineSegment<D, T>(vertices[i], vertices[i + 1]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool 
{
  return areCCW(vertices[0], vertices[1], p) && areCCW(vertices[1], vertices[2], p) &&
         areCCW(vertices[2], vertices[3], p) && areCCW(vertices[3], vertices[0], p);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::area() const noexcept -> T
{
  static_assert(D == 2, "Area of quadrilateral is only defined in 2D");
  // (v2 - v0).cross(v3 - v1) / 2
  Vec<D, T> ac;
  Vec<D, T> bd;
  for (Size i = 0; i < D; ++i) {
    ac[i] = vertices[2][i] - vertices[0][i];
    bd[i] = vertices[3][i] - vertices[1][i];
  }
  return ac.cross(bd) / 2;
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
{
  static_assert(D == 2, "Centroid of quadrilateral is only defined in 2D");
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  Vec<D, T> ab;
  Vec<D, T> ac;
  Vec<D, T> ad;
  for (Size i = 0; i < D; ++i) {
    ab[i] = vertices[1][i] - vertices[0][i];
    ac[i] = vertices[2][i] - vertices[0][i];
    ad[i] = vertices[3][i] - vertices[0][i];
  }
  // Compute the area of each triangle
  T const a1 = ab.cross(ac);
  T const a2 = ac.cross(ad);
  T const a12 = a1 + a2;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    T const v02 = vertices[0][i] + vertices[2][i];
    result[i] = a1 * vertices[1][i] + a2 * vertices[3][i] + a12 * v02;
  }
  return result /= (3 * a12);
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(vertices);
}

} // namespace um2
