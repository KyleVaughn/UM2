namespace um2
{

// -----------------------------------------------------------------------------
// area
// -----------------------------------------------------------------------------
template <typename T>
PURE HOSTDEV constexpr auto
triangleArea(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2) noexcept
    -> T
{
  return cross2(p1 - p0, p2 - p0) / 2;
}

template <typename T>
PURE HOSTDEV constexpr auto
quadrilateralArea(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2,
                  Point2<T> const & p3) noexcept -> T
{
  return cross2(p2 - p0, p3 - p1) / 2;
}

// Area of a linear, planar polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
area(LinearPolygon<N, 2, T> const & poly) noexcept -> T
{
  if constexpr (N == 3) {
    return triangleArea(poly[0], poly[1], poly[2]);
  } else if constexpr (N == 4) {
    return quadrilateralArea(poly[0], poly[1], poly[2], poly[3]);
  } else {
    // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
    // p_n = p_0
    T sum = cross2(poly[N - 1], poly[0]); // cross(p_{n-1}, p_0), the last term
    for (Size i = 0; i < N - 1; ++i) {
      sum += cross2(poly[i], poly[i + 1]);
    }
    return sum / 2;
  }
}

template <typename T>
PURE HOSTDEV constexpr auto
area(Triangle<3, T> const & tri) noexcept -> T
{
  return (tri[1] - tri[0]).cross(tri[2] - tri[0]).norm() / 2;
}

// -----------------------------------------------------------------------------
// centroid
// -----------------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
triangleCentroid(Point<D, T> const & p0, Point<D, T> const & p1,
                 Point<D, T> const & p2) noexcept -> Point<D, T>
{
  return (p0 + p1 + p2) / 3;
}

template <typename T>
PURE HOSTDEV constexpr auto
quadrilateralCentroid(Point2<T> const & p0, Point2<T> const & p1, Point2<T> const & p2,
                      Point2<T> const & p3) noexcept -> Point2<T>
{
  T const a1 = cross2(p1 - p0, p2 - p0);
  T const a2 = cross2(p2 - p0, p3 - p0);
  return (a1 * (p0 + p1 + p2) + a2 * (p0 + p2 + p3)) / (3 * (a1 + a2));
}

// Centroid of a linear, planar polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
centroid(LinearPolygon<N, 2, T> const & poly) noexcept -> Point2<T>
{
  if constexpr (N == 3) {
    return triangleCentroid(poly[0], poly[1], poly[2]);
  } else if constexpr (N == 4) {
    return quadrilateralCentroid(poly[0], poly[1], poly[2], poly[3]);
  } else {
    // Similar to the shoelace formula.
    // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
    T area_sum = cross2(poly[N - 1], poly[0]); // p_{n-1} x p_0, the last term
    Point2<T> centroid_sum = area_sum * (poly[N - 1] + poly[0]);
    for (Size i = 0; i < N - 1; ++i) {
      T const this_area = cross2(poly[i], poly[i + 1]);
      area_sum += this_area;
      centroid_sum += this_area * (poly[i] + poly[i + 1]);
    }
    return centroid_sum / (3 * area_sum);
  }
}

template <typename T>
PURE HOSTDEV constexpr auto
centroid(Triangle<3, T> const & tri) noexcept -> Point3<T>
{
  return triangleCentroid(tri[0], tri[1], tri[2]);
}

// -----------------------------------------------------------------------------
// boundingBox
// -----------------------------------------------------------------------------
// The bounding box of any linear polytope is simply the min and max of its
// vertices.
template <Size K, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
boundingBox(Polytope<K, 1, N, D, T> const & poly) -> AxisAlignedBox<D, T>
{
  return boundingBox(poly.vertices);
}

} // namespace um2
