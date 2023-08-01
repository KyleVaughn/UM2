namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return vertices.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return fv.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadraticTriMesh<D, T, I>::face(Size i) const noexcept -> Face
{
  auto const v0 = static_cast<Size>(fv[i][0]);
  auto const v1 = static_cast<Size>(fv[i][1]);
  auto const v2 = static_cast<Size>(fv[i][2]);
  auto const v3 = static_cast<Size>(fv[i][3]);
  auto const v4 = static_cast<Size>(fv[i][4]);
  auto const v5 = static_cast<Size>(fv[i][5]);
  return QuadraticTriangle<D, T>(vertices[v0], vertices[v1], vertices[v2], vertices[v3],
                                 vertices[v4], vertices[v5]);
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadraticTriMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  AxisAlignedBox<D, T> box = face(0).boundingBox();
  for (Size i = 1; i < numFaces(); ++i) {
    box = um2::boundingBox(box, face(i).boundingBox());
  }
  return box;
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadraticTriMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  static_assert(D == 2, "Only implemented for 2D meshes");
  for (Size i = 0; i < numFaces(); ++i) {
    if (face(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

} // namespace um2
