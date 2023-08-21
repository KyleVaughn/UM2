namespace um2
{

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
getEdge(LinearPolygon<N, D, T> const & p, Size const i) noexcept -> LineSegment<D, T>
{
  assert(0 <= i && i < N);
  return (i < N - 1) ? LineSegment<D, T>(p[i], p[i + 1])
                     : LineSegment<D, T>(p[N - 1], p[0]);
}

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
getEdge(QuadraticPolygon<N, D, T> const & p, Size const i) noexcept
    -> QuadraticSegment<D, T>
{
  assert(0 <= i && i < N);
  constexpr Size m = N / 2;
  return (i < m - 1) ? QuadraticSegment<D, T>(p[i], p[i + 1], p[i + m])
                     : QuadraticSegment<D, T>(p[m - 1], p[0], p[N - 1]);
}

} // namespace um2
