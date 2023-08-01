namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr QuadraticTriangle<D, T>::Polytope(
    Point<D, T> const & p0, Point<D, T> const & p1, Point<D, T> const & p2,
    Point<D, T> const & p3, Point<D, T> const & p4, Point<D, T> const & p5) noexcept
    : v{p0, p1, p2, p3, p4, p5}
{
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  T const ss = static_cast<T>(s);
  // Factoring out the common terms
  T const tt = 1 - rr - ss;
  T const w0 = tt * (2 * tt - 1);
  T const w1 = rr * (2 * rr - 1);
  T const w2 = ss * (2 * ss - 1);
  T const w3 = 4 * rr * tt;
  T const w4 = 4 * rr * ss;
  T const w5 = 4 * ss * tt;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * v[0][i] + w1 * v[1][i] + w2 * v[2][i] + w3 * v[3][i] + w4 * v[4][i] +
                w5 * v[5][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  T const rr = static_cast<T>(4 * r);
  T const ss = static_cast<T>(4 * s);
  T const tt = rr + ss - 3;
  Mat<D, 2, T> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = tt * (v[0][i] - v[3][i]) + (rr - 1) * (v[1][i] - v[3][i]) +
                       ss * (v[4][i] - v[5][i]);
    result.col(1)[i] = tt * (v[0][i] - v[5][i]) + (ss - 1) * (v[2][i] - v[5][i]) +
                       rr * (v[4][i] - v[3][i]);
  }
  return result;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::edge(Size i) const noexcept -> QuadraticSegment<D, T>
{
  assert(i < 3);
  return (i == 2) ? QuadraticSegment<D, T>(v[2], v[0], v[5])
                  : QuadraticSegment<D, T>(v[i], v[i + 1], v[i + 3]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2, "QuadraticTriangle::contains() is only defined for 2D triangles");
  for (Size i = 0; i < 3; ++i) {
    if (!edge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

// -------------------------------------------------------------------
// linearPolygon
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::linearPolygon() const noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(v[0], v[1], v[2]);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::area() const noexcept -> T
{
  static_assert(D == 2, "QuadraticTriangle::area() is only defined for 2D triangles");
  T result = linearPolygon().area();
  for (Size i = 0; i < 3; ++i) {
    result += edge(i).enclosedArea();
  }
  return result;
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::centroid() const noexcept -> Point<D, T>
{
  static_assert(D == 2, "QuadraticTriangle::centroid() is only defined for 2D triangles");
  // By geometric decomposition
  auto const tri = linearPolygon();
  T area_sum = tri.area();
  Point2<T> centroid_sum = area_sum * tri.centroid();
  for (Size i = 0; i < 3; ++i) {
    auto const e = this->edge(i);
    T const a = e.enclosedArea();
    area_sum += a;
    centroid_sum += a * e.enclosedCentroid();
  }
  return centroid_sum / area_sum;
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticTriangle<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  auto result = edge(0).boundingBox();
  result = um2::boundingBox(result, edge(1).boundingBox());
  result = um2::boundingBox(result, edge(2).boundingBox());
  return result;
}

} // namespace um2
