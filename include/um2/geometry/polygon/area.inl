namespace um2
{

template <typename T>
PURE HOSTDEV constexpr auto
area(Triangle2<T> const & tri) noexcept -> T
{
  Vec2<T> const v10 = tri[1] - tri[0];
  Vec2<T> const v20 = tri[2] - tri[0];
  return v10.cross(v20) / 2; // this is the signed area
}

template <typename T>
PURE HOSTDEV constexpr auto
area(Triangle3<T> const & tri) noexcept -> T
{
  Vec3<T> const v10 = tri[1] - tri[0];
  Vec3<T> const v20 = tri[2] - tri[0];
  return v10.cross(v20).norm() / 2; // this is the unsigned area
}

template <typename T>
PURE HOSTDEV constexpr auto
area(Quadrilateral2<T> const & q) noexcept -> T
{
  assert(isConvex(q));
  // (v2 - v0).cross(v3 - v1) / 2
  Vec2<T> const v20 = q[2] - q[0];
  Vec2<T> const v31 = q[3] - q[1];
  return v20.cross(v31) / 2;
}

// Area of a planar linear polygon
template <Size N, typename T>
PURE HOSTDEV constexpr auto
area(PlanarLinearPolygon<N, T> const & p) noexcept -> T
{
  // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
  // p_n = p_0
  T sum = (p[N - 1]).cross(p[0]); // cross(p_{n-1}, p_0), the last term
  for (Size i = 0; i < N - 1; ++i) {
    sum += (p[i]).cross(p[i + 1]);
  }
  return sum / 2;
}

// -------------------------------------------------------------------
// QuadraticPolygon
// -------------------------------------------------------------------

template <Size N, typename T>
PURE HOSTDEV constexpr auto
area(PlanarQuadraticPolygon<N, T> const & q) noexcept -> T
{
  T result = area(linearPolygon(q));
  constexpr Size m = N / 2;
  for (Size i = 0; i < m; ++i) {
    result += enclosedArea(edge(q, i));
  }
  return result;
}

} // namespace um2
