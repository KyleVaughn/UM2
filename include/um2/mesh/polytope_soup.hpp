#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/mesh/element_types.hpp>
//#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

// External dependencies
#include <H5Cpp.h>
#include <pugixml.hpp>

namespace um2
{

//==============================================================================
// POLYTOPE SOUP
//==============================================================================
//// A data structure for storing a mesh, or any collection of polytopes, and data
//// associated with the polytopes. This data structure can be used to:
//// - read/write a mesh and its data from/to a file
//// - convert between mesh data structures
//// - generate subsets
//// - perform mesh operations without assumptions about manifoldness, etc.
////
//////// Note: due to the generality of the data structure, there is effectively a
//////// switch statement in every method. This is not ideal for performance.
//////// See FaceVertexMesh for a more efficient, but less general, data
//////// structure.
////////
//////// _is_morton_sorted: A flag indicating whether the vertices and elements are
////////    sorted using the Morton ordering. This is useful for efficient spatial
////////    queries, but it is not required.
//////// 
//////// _vertices: A list of vertices
////////
//////// _element_types: A list of element types
////////
//////// _element_offsets: A prefix sum of the number of vertices in each element

class PolytopeSoup
{
//
//  bool _is_morton_sorted = false;
  Vector<Point3> _vertices;
  Vector<VTKElemType> _element_types;
  Vector<Int> _element_offsets; // A prefix sum of the number of vertices in each element
  Vector<Int> _element_conn;    // Vertex IDs of each element

  Vector<String> _elset_names;
  Vector<Int> _elset_offsets;      // A prefix sum of the number of elements in each elset
  Vector<Int> _elset_ids;          // Element IDs of each elset (must be sorted)
  Vector<Vector<Float>> _elset_data; // Data associated with each elset

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr PolytopeSoup() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  PolytopeSoup(String const & filename);

  //==============================================================================
  // Methods
  //==============================================================================

  auto
  addElement(VTKElemType type, Vector<Int> const & conn) -> Int;

  auto
  addElset(String const & name, Vector<Int> const & ids, Vector<Float> data = {}) -> Int;

  auto
  addVertex(Float x, Float y, Float z = 0) -> Int;

  auto
  addVertex(Point3 const & p) -> Int;

  PURE [[nodiscard]] auto
  compare(PolytopeSoup const & other) const -> int;

  PURE [[nodiscard]] constexpr auto
  elementConnectivity() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elementOffsets() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elementTypes() const -> Vector<VTKElemType> const &;

  PURE [[nodiscard]] constexpr auto
  elsetNames() const -> Vector<String> const &;

  PURE [[nodiscard]] constexpr auto
  elsetOffsets() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elsetIDs() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elsetData() const -> Vector<Vector<Float>> const &;

//  // If each element has a vertex with the same ID
//  PURE [[nodiscard]] auto
//  elementsShareVertex(Int i, Int j) const -> bool;
//
//  // If each element has a vertex in approximately the same position.
//  // Useful when comparing non-manifold meshes.
//  PURE [[nodiscard]] auto
//  elementsShareVertexApprox(Int i, Int j) const -> bool;
//
  void
  getElement(Int i, VTKElemType & type, Vector<Int> & conn) const;

//  PURE [[nodiscard]] auto
//  getElementBoundingBox(Int i) const -> AxisAlignedBox3;
//
//  PURE [[nodiscard]] auto
//  getElementArea(Int i) const -> Float;
//
//  PURE [[nodiscard]] auto
//  getElementCentroid(Int i) const -> Point3;
//
//  PURE [[nodiscard]] auto
//  getElementMeanChordLength(Int i) const -> Float;

  [[nodiscard]] constexpr auto
  getElemTypes() const -> Vector<VTKElemType>;

  inline void
  getElsetName(Int i, String & name) const;

  void
  getElset(Int i, String & name, Vector<Int> & ids, Vector<Float> & data) const;

  void
  getElset(String const & name, Vector<Int> & ids, Vector<Float> & data) const;

//  // Get the material ID of each element
//  void
//  getMaterialIDs(Vector<MatID> & material_ids,
//                 Vector<String> const & material_names) const;
//
//  void
//  getMaterialNames(Vector<String> & material_names) const;
//
//  PURE [[nodiscard]] constexpr auto
//  getMeshType() const -> MeshType;

  void
  getSubset(String const & elset_name, PolytopeSoup & subset) const;

  PURE [[nodiscard]] constexpr auto
  getVertex(Int i) const -> Point3 const &;

//  // Sort the vertices and elements
//  void
//  mortonSort();
//
//  void
//  mortonSortElements();
//
//  void
//  mortonSortVertices();

  PURE [[nodiscard]] constexpr auto
  numVertices() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElements() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElsets() const -> Int;

  auto
  operator+=(PolytopeSoup const & other) noexcept -> PolytopeSoup &;

  void
  read(String const & filename);

  void
  reserveMoreElements(VTKElemType elem_type, Int num_elems);

  void
  reserveMoreVertices(Int num_verts);

  void
  sortElsets();

////  void
////  translate(Point3 const & v);
////

  PURE [[nodiscard]] constexpr auto
  vertices() const -> Vector<Point3> const &;
  
  void
  write(String const & filename) const;

}; // struct PolytopeSoup

////==============================================================================
//// Free Functions
////==============================================================================
//
//// Get the power of each pin, plate, or other connected subset of elements.
//// Soup must have an elset called "power" with elset data for each element.
//// Assumes a manifold mesh.
//auto
//getPowerRegions(PolytopeSoup const & soup) -> Vector<Pair<Float, Point3>>;
//
//==============================================================================
// Methods
//==============================================================================

PURE constexpr auto
PolytopeSoup::numVertices() const -> Int
{
  return _vertices.size();
}

PURE constexpr auto
PolytopeSoup::numElements() const -> Int
{
  return _element_types.size();
}

PURE constexpr auto
PolytopeSoup::numElsets() const -> Int
{
  return _elset_names.size();
}


PURE constexpr auto
PolytopeSoup::getVertex(Int const i) const -> Point3 const &
{
  ASSERT(i < _vertices.size());
  return _vertices[i];
}

constexpr auto
PolytopeSoup::getElemTypes() const -> Vector<VTKElemType>
{
  Vector<VTKElemType> el_types;
  for (auto const & this_type : _element_types) {
    bool found = false;
    for (auto const & that_type : el_types) {
      if (this_type == that_type) {
        found = true;
        break;
      }
    }
    if (!found) {
      el_types.emplace_back(this_type);
    }
  }
  return el_types;
}

PURE constexpr auto
PolytopeSoup::elementConnectivity() const -> Vector<Int> const &
{
  return _element_conn;
}

PURE constexpr auto
PolytopeSoup::elementOffsets() const -> Vector<Int> const &
{
  return _element_offsets;
}

PURE constexpr auto
PolytopeSoup::elementTypes() const -> Vector<VTKElemType> const &
{
  return _element_types;
}

PURE constexpr auto
PolytopeSoup::elsetNames() const -> Vector<String> const &
{
  return _elset_names;
}

PURE constexpr auto
PolytopeSoup::elsetOffsets() const -> Vector<Int> const &
{
  return _elset_offsets;
}

PURE constexpr auto
PolytopeSoup::elsetIDs() const -> Vector<Int> const &
{
  return _elset_ids;
}

PURE constexpr auto
PolytopeSoup::elsetData() const -> Vector<Vector<Float>> const &
{
  return _elset_data;
}

PURE constexpr auto
PolytopeSoup::vertices() const -> Vector<Point3> const &
{
  return _vertices;
}

void
PolytopeSoup::getElsetName(Int const i, String & name) const
{
  ASSERT(i < _elset_names.size());
  name = _elset_names[i];
}

} // namespace um2
