namespace um2
{

template <Size D, typename T>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticTriangle<D, T> const & q) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(q[0], q[1], q[2]);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
linearPolygon(QuadraticQuadrilateral<D, T> const & q) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(q[0], q[1], q[2], q[3]);
}

} // namespace um2
