#pragma once

#include <um2/common/Log.hpp>
#include <um2/geometry/Point.hpp>
#include <um2/mesh/CellType.hpp>
#include <um2/mesh/MeshType.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>

#include <algorithm>
#ifdef _OPENMP
#  include <parallel/algorithm>
#endif
#include <string>
#include <vector>

namespace um2
{

// -----------------------------------------------------------------------------
// MESH FILE
// -----------------------------------------------------------------------------
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

template <std::floating_point T, std::signed_integral I>
struct MeshFile {

  std::string filepath; // path to the mesh file, including file name
  std::string name;     // name of the mesh

  MeshFileFormat format = MeshFileFormat::None;

  std::vector<T> nodes_x;
  std::vector<T> nodes_y;
  std::vector<T> nodes_z;

  MeshType type = MeshType::None;
  std::vector<I> element_conn;

  // Instead of storing a vector of vector, we store the elset IDs in a single contiguous
  // array. This is much less convenient for adding or deleting elsets, but it is much
  // more efficient for generating submeshes and other more time-critical operations.
  std::vector<std::string> elset_names;
  std::vector<I> elset_offsets; // size = num_elsets + 1
  std::vector<I> elset_ids;     // size = elset_offsets[num_elsets]

  constexpr MeshFile() = default;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  constexpr void
  sortElsets();

  constexpr void
  getSubmesh(std::string const & elset_name, MeshFile<T, I> & submesh) const;
  //  //
  //  //    constexpr MeshType get_mesh_type() const;
  //  //
  //  constexpr void
  //  getMaterialNames(std::vector<std::string> & material_names) const;
  //
  //    constexpr void get_material_ids(std::vector<MaterialID> & material_ids) const;
  //
  //    constexpr void get_material_ids(std::vector<MaterialID> & material_ids,
  //                                    std::vector<std::string> const & material_names)
  //                                    const;

}; // struct MeshFile

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(MeshFile<T, I> const & a, MeshFile<T, I> const & b) -> int;

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareTopology(MeshFile<T, I> const & a, MeshFile<T, I> const & b) -> int;

} // namespace um2

#include "MeshFile.inl"
