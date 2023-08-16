namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
QuadMesh<D, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
  toFaceVertexMesh(file, *this);
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return um2::numVertices(*this);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return um2::numFaces(*this);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
QuadMesh<D, T, I>::getFace(Size i) const noexcept -> Face
{
  return um2::getFace(*this, i);
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
QuadMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  return um2::faceContaining(*this, p);
}

template <Size D, std::floating_point T, std::signed_integral I>
void
QuadMesh<D, T, I>::flipFace(Size i) noexcept
{
  um2::swap(fv[i][1], fv[i][3]);
}

template <Size D, std::floating_point T, std::signed_integral I>
void
QuadMesh<D, T, I>::toMeshFile(MeshFile<T, I> & file) const noexcept
{
  um2::toMeshFile(*this, file);
}

} // namespace um2
