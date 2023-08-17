namespace um2
{

template <typename T>
PURE HOSTDEV constexpr auto
contains(Triangle2<T> const & tri, Point2<T> const & p) noexcept -> bool
{
  // Benchmarking shows it is faster to compute the areCCW() test for each
  // edge, then return based on the AND of the results, rather than compute
  // the areCCW one at a time and return as soon as one is false.
  bool const b0 = areCCW(tri[0], tri[1], p);
  bool const b1 = areCCW(tri[1], tri[2], p);
  bool const b2 = areCCW(tri[2], tri[0], p);
  return b0 && b1 && b2;
}

template <typename T>
PURE HOSTDEV constexpr auto
contains(Quadrilateral2<T> const & tri, Point2<T> const & p) noexcept -> bool
{
  bool const b0 = areCCW(tri[0], tri[1], p);
  bool const b1 = areCCW(tri[1], tri[2], p);
  bool const b2 = areCCW(tri[2], tri[3], p);
  bool const b3 = areCCW(tri[3], tri[0], p);
  return b0 && b1 && b2 && b3;
}

//==============================================================================
// QuadraticPolygon
//==============================================================================

template <Size N, typename T>
PURE HOSTDEV constexpr auto
contains(PlanarQuadraticPolygon<N, T> const & q, Point2<T> const & p) noexcept -> bool
{
  // Benchmarking shows that the opposite conclusion is true for quadratic
  // polygons: it is faster to compute the areCCW() test for each edge, short
  // circuiting as soon as one is false, rather than compute all of them.
  constexpr Size m = N / 2;
  for (Size i = 0; i < m; ++i) {
    if (!getEdge(q, i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

} // namespace um2
