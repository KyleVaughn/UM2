namespace um2
{

//==============================================================================
// numVertices
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numVertices(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.vertices.size();
}

//==============================================================================
// numFaces
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
numFaces(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept -> Size
{
  return mesh.fv.size();
}

} // namespace um2
