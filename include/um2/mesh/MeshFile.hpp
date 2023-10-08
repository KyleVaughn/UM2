#pragma once

#include <um2/common/Log.hpp>
#include <um2/geometry/Point.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>

#include <algorithm>
#include <string>
#include <vector>

#if UM2_USE_TBB
#  include <execution>
#endif

namespace um2
{

//==============================================================================
// MESH FILE
//==============================================================================
//
// An intermediate representation of a mesh and mesh data that can be used to:
// - read/write a mesh and its data from/to a file
// - convert between mesh data structures
// - generate submeshes
// - perform mesh operations without assumptions about manifoldness, etc.
//
// Note: due to the generality of the data structure, there is effectively a
// switch statement in every method that operates on the mesh. This is not ideal for
// performance. See FaceVertexMesh for a more efficient, but less general, data
// structure.

//==============================================================================
// Element topology identifiers
//==============================================================================

enum class XDMFElemType : int8_t {

  // Linear cells
  Triangle = 4,
  Quad = 5,

  // Quadratic, isoparametric cells
  QuadraticTriangle = 36,
  QuadraticQuad = 37

};

enum class MeshType : int8_t {
  None = 0,
  Tri = 3,
  Quad = 4,
  TriQuad = 7,
  QuadraticTri = 6,
  QuadraticQuad = 8,
  QuadraticTriQuad = 14
};

constexpr auto
verticesPerCell(MeshType const type) -> Size
{
  switch (type) {
  case MeshType::Tri:
    return 3;
  case MeshType::Quad:
    return 4;
  case MeshType::QuadraticTri:
    return 6;
  case MeshType::QuadraticQuad:
    return 8;
  default:
    assert(false);
    return -1;
  }
}

constexpr auto
xdmfElemTypeToMeshType(int8_t x) -> MeshType
{
  switch (x) {
  case static_cast<int8_t>(XDMFElemType::Triangle):
    return MeshType::Tri;
  case static_cast<int8_t>(XDMFElemType::Quad):
    return MeshType::Quad;
  case static_cast<int8_t>(XDMFElemType::QuadraticTriangle):
    return MeshType::QuadraticTri;
  case static_cast<int8_t>(XDMFElemType::QuadraticQuad):
    return MeshType::QuadraticQuad;
  default:
    assert(false);
    return MeshType::None;
  }
}

constexpr auto
meshTypeToXDMFElemType(MeshType x) -> int8_t
{
  switch (x) {
  case MeshType::Tri:
    return static_cast<int8_t>(XDMFElemType::Triangle);
  case MeshType::Quad:
    return static_cast<int8_t>(XDMFElemType::Quad);
  case MeshType::QuadraticTri:
    return static_cast<int8_t>(XDMFElemType::QuadraticTriangle);
  case MeshType::QuadraticQuad:
    return static_cast<int8_t>(XDMFElemType::QuadraticQuad);
  default:
    assert(false);
    return -1;
  }
}

template <std::floating_point T, std::signed_integral I>
struct MeshFile {

  std::string filepath; // full path to the mesh file, including the filename
  std::string name;     // name of the mesh (not necessarily the same as the filename)

  std::vector<Point3<T>> vertices;
  std::vector<MeshType> element_types;
  std::vector<I> element_offsets; // size = num_cells + 1
  std::vector<I> element_conn;

  // Instead of storing a vector of vector, we store the elset IDs in a single contiguous
  // array. This is much less convenient for adding or deleting elsets, but it is much
  // more efficient for generating submeshes and other more time-critical operations.
  std::vector<std::string> elset_names;
  std::vector<I> elset_offsets; // size = num_elsets + 1
  std::vector<I> elset_ids;     // size = elset_offsets[num_elsets]

  // Face data sets (optional)
  std::vector<std::string> face_data_names;
  std::vector<std::vector<T>> face_data; // each std::vector<T> has size = num_faces

  constexpr MeshFile() = default;

  //==============================================================================
  // Methods
  //==============================================================================

  PURE [[nodiscard]] constexpr auto
  numCells() const -> size_t;

  PURE [[nodiscard]] constexpr auto
  getMeshType() const -> MeshType;

  constexpr void
  sortElsets();

  void
  getSubmesh(std::string const & elset_name, MeshFile<T, I> & submesh) const;

  void
  getMaterialNames(std::vector<std::string> & material_names) const;

  constexpr void
  getMaterialIDs(std::vector<MaterialID> & material_ids,
                 std::vector<std::string> const & material_names) const;

}; // struct MeshFile

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(MeshFile<T, I> const & lhs, MeshFile<T, I> const & rhs) -> int;

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareTopology(MeshFile<T, I> const & lhs, MeshFile<T, I> const & rhs) -> int;

} // namespace um2

#include "MeshFile.inl"
