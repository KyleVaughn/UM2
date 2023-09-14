// Free functions
#include <um2/mesh/face_vertex_mesh/intersect.inl>
#include <um2/mesh/face_vertex_mesh/toFaceVertexMesh.inl>
#include <um2/mesh/face_vertex_mesh/toMeshFile.inl>
#include <um2/mesh/face_vertex_mesh/validateMesh.inl>
namespace um2
{

//==============================================================================
//==============================================================================
// Free functions
//==============================================================================
//==============================================================================

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

//==============================================================================
// TriMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(TriMesh<D, T, I> const & mesh, Size i) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                        mesh.vertices[static_cast<Size>(mesh.fv[i][2])]);
}

//==============================================================================
// QuadMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadMesh<D, T, I> const & mesh, Size i) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                             mesh.vertices[static_cast<Size>(mesh.fv[i][3])]);
}

//==============================================================================
// QuadraticTriMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticTriMesh<D, T, I> const & mesh, Size i) noexcept
    -> QuadraticTriangle<D, T>
{
  return QuadraticTriangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][3])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][4])],
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][5])]);
}

//==============================================================================
// QuadraticQuadMesh.getFace
//==============================================================================

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticQuadMesh<D, T, I> const & mesh, Size i) noexcept
    -> QuadraticQuadrilateral<D, T>
{
  return QuadraticQuadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][1])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][2])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][3])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][4])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][5])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][6])],
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][7])]);
}

//==============================================================================
// boundingBox
//==============================================================================

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

//==============================================================================
//==============================================================================
// Member functions
//==============================================================================
//==============================================================================

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

//==============================================================================
// faceContaining
//==============================================================================

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
