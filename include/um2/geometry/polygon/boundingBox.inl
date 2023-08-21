namespace um2
{

template <Size N, typename T>
PURE HOSTDEV constexpr auto
boundingBox(PlanarQuadraticPolygon<N, T> const & p) noexcept -> AxisAlignedBox2<T>
{
  AxisAlignedBox2<T> box = boundingBox(getEdge(p, 0));
  Size const num_edges = polygonNumEdges<2, N>();
  for (Size i = 1; i < num_edges; ++i) {
    box += boundingBox(getEdge(p, i));
  }
  return box;
}

} // namespace um2
