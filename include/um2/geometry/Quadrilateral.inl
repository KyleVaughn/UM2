namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr Quadrilateral<D, T>::Polytope(Point<D, T> const & p0,
                                                Point<D, T> const & p1,
                                                Point<D, T> const & p2,
                                                Point<D, T> const & p3) noexcept
    : v{p0, p1, p2, p3}
{
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

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
    result[i] = w0 * v[0][i] + w1 * v[1][i] + w2 * v[2][i] + w3 * v[3][i];
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
    jac(i, 0) = w0 * (v[1][i] - v[0][i]) - ss * (v[3][i] - v[2][i]);
    jac(i, 1) = w2 * (v[3][i] - v[0][i]) - rr * (v[1][i] - v[2][i]);
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
  return (i == 3) ? LineSegment<D, T>(v[3], v[0]) : LineSegment<D, T>(v[i], v[i + 1]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  // GPU alternative? Maybe do all the computations up until the final comparison and
  // assign the value to a bool variable. Then return the bool variable.
  return areCCW(v[0], v[1], p) && areCCW(v[1], v[2], p) && areCCW(v[2], v[3], p) &&
         areCCW(v[3], v[0], p);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::area() const noexcept -> T
{
  static_assert(D == 2, "Area of quadrilateral is only defined in 2D");
  assert(isConvex());
  // (v2 - v0).cross(v3 - v1) / 2
  Vec<D, T> v20 = v[2] - v[0];
  Vec<D, T> v31 = v[3] - v[1];
  return v20.cross(v31) / 2;
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
{
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  static_assert(D == 2, "Centroid of quadrilateral is only defined in 2D");
  assert(isConvex());
  // If the quadrilateral is not convex, then we need to choose the correct
  // two triangles to decompose the quadrilateral into. If the quadrilateral
  // is convex, any two triangles will do.
  Vec<D, T> v10 = v[1] - v[0];
  Vec<D, T> v20 = v[2] - v[0];
  Vec<D, T> v30 = v[3] - v[0];
  // Compute the area of each triangle
  T const a1 = v10.cross(v20);
  T const a2 = v20.cross(v30);
  T const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = a1 * v[1][i] + a2 * v[3][i] + a12 * (v[0][i] + v[2][i]);
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
  return um2::boundingBox(v);
}

// -------------------------------------------------------------------
// isConvex
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Quadrilateral<D, T>::isConvex() const noexcept -> bool
{
  static_assert(D == 2, "Convexity of quadrilateral is only defined in 2D");
  // Alternative: Use sum of areas of triangles
  return areCCW(v[0], v[1], v[2]) && areCCW(v[1], v[2], v[3]) &&
         areCCW(v[2], v[3], v[0]) && areCCW(v[3], v[0], v[1]);
}
} // namespace um2
