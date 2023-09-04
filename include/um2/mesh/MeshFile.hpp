#pragma once

#include <um2/common/Log.hpp>
#include <um2/geometry/Point.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>

#include <algorithm>
#ifdef UM2_USE_OPENMP
#  include <parallel/algorithm>
#endif
#include <string>
#include <vector>

namespace um2
{

//==============================================================================
// MESH FILE
//==============================================================================
// An intermediate representation of a mesh that can be used to:
// - read a mesh from a file
// - write a mesh to a file
// - convert a mesh to another format
//

enum class MeshFileFormat : int8_t {
  None = 0,
  Abaqus = 1,
  XDMF = 2,
};

enum class VTKCellType : int8_t {

  // Linear cells
  Triangle = 5,
  Quad = 9,

  // Quadratic, isoparametric cells
  QuadraticTriangle = 22,
  QuadraticQuad = 23

};

enum class AbaqusCellType : int8_t {

  // Linear cells
  CPS3 = static_cast<int8_t>(VTKCellType::Triangle),
  CPS4 = static_cast<int8_t>(VTKCellType::Quad),

  // Quadratic, isoparametric cells
  CPS6 = static_cast<int8_t>(VTKCellType::QuadraticTriangle),
  CPS8 = static_cast<int8_t>(VTKCellType::QuadraticQuad)

};

enum class XDMFCellType : int8_t {

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
xdmfCellTypeToMeshType(int8_t x) -> MeshType
{
  switch (x) {
  case static_cast<int8_t>(XDMFCellType::Triangle):
    return MeshType::Tri;
  case static_cast<int8_t>(XDMFCellType::Quad):
    return MeshType::Quad;
  case static_cast<int8_t>(XDMFCellType::QuadraticTriangle):
    return MeshType::QuadraticTri;
  case static_cast<int8_t>(XDMFCellType::QuadraticQuad):
    return MeshType::QuadraticQuad;
  default:
    assert(false);
    return MeshType::None;
  }
}

constexpr auto
meshTypeToXDMFCellType(MeshType x) -> int8_t
{
  switch (x) {
  case MeshType::Tri:
    return static_cast<int8_t>(XDMFCellType::Triangle);
  case MeshType::Quad:
    return static_cast<int8_t>(XDMFCellType::Quad);
  case MeshType::QuadraticTri:
    return static_cast<int8_t>(XDMFCellType::QuadraticTriangle);
  case MeshType::QuadraticQuad:
    return static_cast<int8_t>(XDMFCellType::QuadraticQuad);
  default:
    assert(false);
    return -1;
  }
}

template <std::floating_point T, std::signed_integral I>
struct MeshFile {

  std::string filepath; // path to the mesh file, including file name
  std::string name;     // name of the mesh

  MeshFileFormat format = MeshFileFormat::None;

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

  constexpr void
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
