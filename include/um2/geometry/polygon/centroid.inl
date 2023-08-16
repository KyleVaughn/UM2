namespace um2
{

template <typename T>
PURE HOSTDEV constexpr auto
centroid(Triangle2<T> const & tri) noexcept -> Point2<T>
{
  // (v0 + v1 + v2) / 3
  Point2<T> result;
  for (Size i = 0; i < 2; ++i) {
    result[i] = tri[0][i] + tri[1][i] + tri[2][i];
  }
  return result /= 3;
}

template <typename T>
PURE HOSTDEV constexpr auto
centroid(Triangle3<T> const & tri) noexcept -> Point3<T>
{
  // (v0 + v1 + v2) / 3
  Point3<T> result;
  for (Size i = 0; i < 3; ++i) {
    result[i] = tri[0][i] + tri[1][i] + tri[2][i];
  }
  return result /= 3;
}

template <typename T>
PURE HOSTDEV constexpr auto
centroid(Quadrilateral2<T> const & quad) noexcept -> Point2<T>
{
  // Algorithm: Decompose the quadrilateral into two triangles and
  // compute the centroid of each triangle. The centroid of the
  // quadrilateral is the weighted average of the centroids of the
  // two triangles, where the weights are the areas of the triangles.
  assert(isConvex(quad));
  // If the quadrilateral is not convex, then we need to choose the correct
  // two triangles to decompose the quadrilateral into. If the quadrilateral
  // is convex, any two triangles will do.
  Vec2<T> const v10 = quad[1] - quad[0];
  Vec2<T> const v20 = quad[2] - quad[0];
  Vec2<T> const v30 = quad[3] - quad[0];
  // Compute the area of each triangle
  T const a1 = v10.cross(v20);
  T const a2 = v20.cross(v30);
  T const a12 = a1 + a2;
  // Compute the centroid of each triangle
  // (v0 + v1 + v2) / 3
  // Each triangle shares v0 and v2, so we factor out the common terms
  Point2<T> result;
  for (Size i = 0; i < 2; ++i) {
    result[i] = a1 * quad[1][i] + a2 * quad[3][i] + a12 * (quad[0][i] + quad[2][i]);
  }
  return result /= (3 * a12);
}

// Centroid of a planar linear polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
centroid(PlanarLinearPolygon<N, T> const & p) noexcept -> Point2<T>
{
  // Similar to the shoelace formula.
  // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
  T area_sum = (p[N - 1]).cross(p[0]); // p_{n-1} x p_0, the last term
  Point2<T> centroid_sum = area_sum * (p[N - 1] + p[0]);
  for (Size i = 0; i < N - 1; ++i) {
    T const a = (p[i]).cross(p[i + 1]);
    area_sum += a;
    centroid_sum += a * (p[i] + p[i + 1]);
  }
  return centroid_sum / (static_cast<T>(3) * area_sum);
}

// -------------------------------------------------------------------
// QuadraticPolygon
// -------------------------------------------------------------------

template <Size N, typename T>
PURE HOSTDEV constexpr auto
centroid(PlanarQuadraticPolygon<N, T> const & q) noexcept -> Point2<T>
{
  auto lin_poly = linearPolygon(q);
  T area_sum = lin_poly.area();
  Point2<T> centroid_sum = area_sum * centroid(lin_poly);
  constexpr Size m = N / 2;
  for (Size i = 0; i < m; ++i) {
    auto const e = getEdge(q, i);
    T const a = enclosedArea(e);
    area_sum += a;
    centroid_sum += a * enclosedCentroid(e);
  }
  return centroid_sum / area_sum; 
}

} // namespace um2
