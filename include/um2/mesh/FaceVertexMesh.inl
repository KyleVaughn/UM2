// Free functions
#include <um2/mesh/face_vertex_mesh/boundingBox.inl>
#include <um2/mesh/face_vertex_mesh/faceContaining.inl>
#include <um2/mesh/face_vertex_mesh/getFace.inl>
#include <um2/mesh/face_vertex_mesh/intersect.inl>
#include <um2/mesh/face_vertex_mesh/numVerticesFaces.inl>
#include <um2/mesh/face_vertex_mesh/toFaceVertexMesh.inl>
#include <um2/mesh/face_vertex_mesh/toMeshFile.inl>
#include <um2/mesh/face_vertex_mesh/validateMesh.inl>

// Member functions
namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
FaceVertexMesh<P, N, D, T, I>::FaceVertexMesh(MeshFile<T, I> const & file)
{
  um2::toFaceVertexMesh(file, *this);
}

//==============================================================================
// numVertices
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numVertices() const noexcept -> Size
{
  return um2::numVertices(*this);
}

//==============================================================================
// numFaces
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::numFaces() const noexcept -> Size
{
  return um2::numFaces(*this);
}

//==============================================================================
// getFace
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::getFace(Size i) const noexcept -> Face
{
  return um2::getFace(*this, i);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// faceContaining
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
PURE [[nodiscard]] constexpr auto
FaceVertexMesh<P, N, D, T, I>::faceContaining(Point<D, T> const & p) const noexcept
    -> Size
{
  return um2::faceContaining(*this, p);
}

//==============================================================================
// flipFace
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::flipFace(Size i) noexcept
{
  if constexpr (P == 1 && N == 3) {
    um2::swap(fv[i][1], fv[i][2]);
  } else if constexpr (P == 1 && N == 4) {
    um2::swap(fv[i][1], fv[i][3]);
  } else if constexpr (P == 2 && N == 6) {
    um2::swap(fv[i][1], fv[i][2]);
    um2::swap(fv[i][3], fv[i][5]);
  } else if constexpr (P == 2 && N == 8) {
    um2::swap(fv[i][1], fv[i][3]);
    um2::swap(fv[i][4], fv[i][7]);
  }
}

//==============================================================================
// toMeshFile
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::toMeshFile(MeshFile<T, I> & file) const noexcept
{
  um2::toMeshFile(*this, file);
}

//==============================================================================
// getFaceAreas
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
constexpr auto
FaceVertexMesh<P, N, D, T, I>::getFaceAreas() const noexcept -> Vector<T>
{
  Vector<T> areas(numFaces());
  for (Size i = 0; i < numFaces(); ++i) {
    areas[i] = getFace(i).area();
  }
  return areas;
}

//==============================================================================
// intersect
//==============================================================================

template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
void
FaceVertexMesh<P, N, D, T, I>::intersect(Ray<D, T> const & ray, T * intersections,
                                         Size * const n) const noexcept
{
  um2::intersect(*this, ray, intersections, n);
}

} // namespace um2
