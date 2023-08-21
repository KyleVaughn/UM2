namespace um2
{

template <Size P, Size N, std::floating_point T, std::signed_integral I>
PURE constexpr auto
faceContaining(PlanarPolygonMesh<P, N, T, I> const & mesh, Point2<T> const & p) noexcept
    -> Size
{
  for (Size i = 0; i < numFaces(mesh); ++i) {
    if (mesh.getFace(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

} // namespace um2
