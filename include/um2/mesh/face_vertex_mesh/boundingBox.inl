namespace um2
{

template <Size N, Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
boundingBox(LinearPolygonMesh<N, D, T, I> const & mesh) noexcept -> AxisAlignedBox<D, T>
{
  return boundingBox(mesh.vertices);
}

template <Size N, Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
boundingBox(QuadraticPolygonMesh<N, D, T, I> const & mesh) noexcept
    -> AxisAlignedBox<D, T>
{
  AxisAlignedBox<D, T> box = mesh.getFace(0).boundingBox();
  for (Size i = 1; i < numFaces(mesh); ++i) {
    box += mesh.getFace(i).boundingBox();
  }
  return box;
}

} // namespace um2
