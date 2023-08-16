namespace um2
{

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
TriMesh<D, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
  toFaceVertexMesh(file, *this);
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return um2::numVertices(*this);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return um2::numFaces(*this); 
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::getFace(Size i) const noexcept -> Face
{
  return um2::getFace(*this, i); 
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
TriMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
TriMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  return um2::faceContaining(*this, p); 
}

template <Size D, std::floating_point T, std::signed_integral I>
void
TriMesh<D, T, I>::flipFace(Size i) noexcept
{
  um2::swap(fv[i][1], fv[i][2]); 
}

template <Size D, std::floating_point T, std::signed_integral I>
void
TriMesh<D, T, I>::toMeshFile(MeshFile<T, I> & file) const noexcept
{
  toMeshFile(*this, file);
}

} // namespace um2
