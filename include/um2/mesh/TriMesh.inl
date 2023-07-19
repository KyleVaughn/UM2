namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return vertices.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return fv.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::face(Size i) const noexcept -> Face
{
  auto const v0 = static_cast<Size>(fv[i][0]);
  auto const v1 = static_cast<Size>(fv[i][1]);
  auto const v2 = static_cast<Size>(fv[i][2]);
  return Triangle<D, T>(vertices[v0], vertices[v1], vertices[v2]);
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOST constexpr auto
TriMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T> 
{
  return um2::boundingBox(vertices); 
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOST constexpr auto
TriMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  static_assert(D==2, "Only implemented for 2D meshes");
  for (Size i = 0; i < numFaces(); ++i) {
    if (face(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1; 
}

} // namespace um2
