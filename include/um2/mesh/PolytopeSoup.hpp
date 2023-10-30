#pragma once

#include <um2/common/Log.hpp>
#include <um2/geometry/Point.hpp>
#include <um2/stdlib/String.hpp>
#include <um2/stdlib/Vector.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>

#if UM2_USE_TBB
#  include <execution>
#endif

namespace um2
{

//==============================================================================
// POLYTOPE SOUP
//==============================================================================
//
// A data structure for storing a mesh, or any collection of polytopes, and data
// associated with the polytopes. This data structure can be used to:
// - read/write a mesh and its data from/to a file
// - convert between mesh data structures
// - generate submeshes
// - perform mesh operations without assumptions about manifoldness, etc.
//
// Note: due to the generality of the data structure, there is effectively a
// switch statement in every method. This is not ideal for performance.
// See FaceVertexMesh for a more efficient, but less general, data
// structure.

//==============================================================================
// Element topology identifiers
//==============================================================================

enum class VTKElemType : int8_t {
  None = 0,
  Vertex = 1,
  Line = 3,
  Triangle = 5,
  Quad = 9,
  QuadraticEdge = 21,
  QuadraticTriangle = 22,
  QuadraticQuad = 23
};

enum class XDMFElemType : int8_t {
  None = 0,
  Vertex = 1,
  Line = 2,
  Triangle = 4,
  Quad = 5,
  QuadraticEdge = 34,
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
verticesPerElem(VTKElemType const type) -> Size
{
  switch (type) {
  case VTKElemType::Vertex:
    return 1;
  case VTKElemType::Line:
    return 2;
  case VTKElemType::Triangle:
    return 3;
  case VTKElemType::Quad:
    return 4;
  case VTKElemType::QuadraticEdge:
    return 3;
  case VTKElemType::QuadraticTriangle:
    return 6;
  case VTKElemType::QuadraticQuad:
    return 8;
  default:
    assert(false);
    return -1;
  }
}

constexpr auto
xdmfToVTKElemType(int8_t x) -> VTKElemType
{
  switch (x) {
  case static_cast<int8_t>(XDMFElemType::Vertex):
    return VTKElemType::Vertex;
  case static_cast<int8_t>(XDMFElemType::Line):
    return VTKElemType::Line;
  case static_cast<int8_t>(XDMFElemType::Triangle):
    return VTKElemType::Triangle;
  case static_cast<int8_t>(XDMFElemType::Quad):
    return VTKElemType::Quad;
  case static_cast<int8_t>(XDMFElemType::QuadraticEdge):
    return VTKElemType::QuadraticEdge;
  case static_cast<int8_t>(XDMFElemType::QuadraticTriangle):
    return VTKElemType::QuadraticTriangle;
  case static_cast<int8_t>(XDMFElemType::QuadraticQuad):
    return VTKElemType::QuadraticQuad;
  default:
    assert(false);
    return VTKElemType::None;
  }
}

constexpr auto
vtkToXDMFElemType(VTKElemType x) -> int8_t
{
  switch (x) {
  case VTKElemType::Vertex:
    return static_cast<int8_t>(XDMFElemType::Vertex);
  case VTKElemType::Line:
    return static_cast<int8_t>(XDMFElemType::Line);
  case VTKElemType::Triangle:
    return static_cast<int8_t>(XDMFElemType::Triangle);
  case VTKElemType::Quad:
    return static_cast<int8_t>(XDMFElemType::Quad);
  case VTKElemType::QuadraticEdge:
    return static_cast<int8_t>(XDMFElemType::QuadraticEdge);
  case VTKElemType::QuadraticTriangle:
    return static_cast<int8_t>(XDMFElemType::QuadraticTriangle);
  case VTKElemType::QuadraticQuad:
    return static_cast<int8_t>(XDMFElemType::QuadraticQuad);
  default:
    assert(false);
    return -1;
  }
}

template <std::floating_point T, std::signed_integral I>
struct PolytopeSoup {

  Vector<Point3<T>> vertices;
  Vector<VTKElemType> element_types;
  Vector<I> element_offsets; // A prefix sum of the number of vertices in each element
  Vector<I> element_conn;    // Vertex IDs of each element

  // Instead of storing a vector of vector, we store the elset IDs in a single contiguous
  // array. This is much less convenient for adding or deleting elsets, but it is much
  // more efficient for generating submeshes and other more time-critical operations.
  Vector<String> elset_names;
  Vector<I> elset_offsets;      // A prefix sum of the number of elements in each elset
  Vector<I> elset_ids;          // Element IDs of each elset
  Vector<Vector<T>> elset_data; // Data associated with each elset

  constexpr PolytopeSoup() = default;

  //==============================================================================
  // Methods
  //==============================================================================

  PURE [[nodiscard]] constexpr auto
  numElems() const -> Size;

  PURE [[nodiscard]] constexpr auto
  hasElsetData() const -> bool;

  PURE [[nodiscard]] constexpr auto
  getElemTypes() const -> Vec<8, VTKElemType>;

  PURE [[nodiscard]] constexpr auto
  getMeshType() const -> MeshType;

  constexpr void
  addElset(String const & name, Vector<I> const & ids, Vector<T> data = {});

  constexpr void
  sortElsets();

  void
  getMaterialNames(Vector<String> & material_names) const;
  //
  //  void
  //  getSubmesh(std::string const & elset_name, PolytopeSoup<T, I> & submesh) const;
  //
  //  constexpr void
  //  getMaterialIDs(std::vector<MaterialID> & material_ids,
  //                 std::vector<std::string> const & material_names) const;
  //
}; // struct PolytopeSoup

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(PolytopeSoup<T, I> const & lhs, PolytopeSoup<T, I> const & rhs) -> int;

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareTopology(PolytopeSoup<T, I> const & lhs, PolytopeSoup<T, I> const & rhs) -> int;

} // namespace um2

#include "PolytopeSoup.inl"
