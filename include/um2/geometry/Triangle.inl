namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr Triangle<D, T>::Polytope(Point<D, T> const & p0,
                                           Point<D, T> const & p1,
                                           Point<D, T> const & p2) noexcept
  : v{p0, p1, p2}
{}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Triangle<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  // (1 - r - s) v0 + r v1 + s v2
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  T const w0 = (1 - rr - ss);
  // T const w1 = rr;
  // T const w2 = ss;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * v[0][i] + rr * v[1][i] + ss * v[2][i];
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
  return Mat<D, 2, T>(v[1] - v[0], v[2] - v[0]);
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::edge(Size i) const noexcept -> LineSegment<D, T>
{
  assert(i < 3);
  return (i == 2) ? LineSegment<D, T>(v[2], v[0]) : LineSegment<D, T>(v[i], v[i + 1]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2, "Triangle::contains() is only defined for 2D triangles");
  // NOLINTBEGIN(readability-identifier-naming)
  // P = V0 + r(V1 - V0) + s(V2 - V0)
  // P - V0 = r(V1 - V0) + s(V2 - V0)
  // Let A = V1 - V0, B = V2 - V0, C = P - V0
  // C = rA + sB = [A B] [r s]^T
  // Using Cramer's rule
  // r = det([C B]) / det([A B])
  // s = det([A C]) / det([A B])
  // Note that det([A B]) = A x B
  Vec<D, T> const A = v[1] - v[0];
  Vec<D, T> const B = v[2] - v[0];
  Vec<D, T> const C = p - v[0];
  T const invdetAB = 1 / A.cross(B);
  T const r = C.cross(B) * invdetAB;
  T const s = A.cross(C) * invdetAB;
  return (r >= 0) && (s >= 0) && (r + s <= 1);
  // NOLINTEND(readability-identifier-naming)

  // GPU alternative? Maybe do all the computations up until the final comparison and
  // assign the value to a bool variable. Then return the bool variable.
  // return areCCW(v[0], v[1], p) && areCCW(v[1], v[2], p) && areCCW(v[2], v[0], p)
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
Triangle<D, T>::area() const noexcept -> T
{
  // return (v1 - v0).cross(v2 - v0).norm() / 2;
  Vec<D, T> v10 = v[1] - v[0];
  Vec<D, T> v20 = v[2] - v[0];
  if constexpr (D == 2) {
    return v10.cross(v20) / 2; // this is the signed area
  } else if constexpr (D == 3) {
    return v10.cross(v20).norm() / 2; // this is the unsigned area
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
  // (v0 + v1 + v2) / 3
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = v[0][i] + v[1][i] + v[2][i];
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
  return um2::boundingBox(v);
}

} // namespace um2
