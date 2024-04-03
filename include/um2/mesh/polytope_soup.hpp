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
//// - generate subsetes
//// - perform mesh operations without assumptions about manifoldness, etc.
////
//// Note: due to the generality of the data structure, there is effectively a
//// switch statement in every method. This is not ideal for performance.
//// See FaceVertexMesh for a more efficient, but less general, data
//// structure.
////
//// _is_morton_sorted: A flag indicating whether the vertices and elements are
////    sorted using the Morton ordering. This is useful for efficient spatial
////    queries, but it is not required.
//// 
//// _vertices: A list of vertices
////
//// _element_types: A list of element types
////
//// _element_offsets: A prefix sum of the number of vertices in each element

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

  //==============================================================================
  // Private Methods
  //==============================================================================

  void
  writeXDMF(String const & filepath) const;

  void
  writeXDMFUniformGrid(String const & name, //Vector<String> const & material_names,
                       pugi::xml_node & xdomain, H5::H5File & h5file,
                       String const & h5filename, String const & h5path) const;

  void
  writeXDMFGeometry(pugi::xml_node & xgrid, H5::Group & h5group,
                    String const & h5filename, String const & h5path) const;

  void
  writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group,
                    String const & h5filename, String const & h5path) const;

  void
  writeXDMFElsets(pugi::xml_node & xgrid, H5::Group & h5group, String const & h5filename,
                  String const & h5path) const; //Vector<String> const & material_names) const;

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
//
//  void
//  getSubset(String const & elset_name, PolytopeSoup & subset) const;
//
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
  numVerts() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElems() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElsets() const -> Int;

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
PolytopeSoup::numVerts() const -> Int
{
  return _vertices.size();
}

PURE constexpr auto
PolytopeSoup::numElems() const -> Int
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

//PURE constexpr auto
//PolytopeSoup::getMeshType() const -> MeshType
//{
//  // Loop through the element types to determine which 1 or 2 mesh types are
//  // present.
//  VTKElemType type1 = VTKElemType::None;
//  VTKElemType type2 = VTKElemType::None;
//  for (auto const & this_type : _element_types) {
//    if (type1 == VTKElemType::None) {
//      type1 = this_type;
//    }
//    if (type1 == this_type) {
//      continue;
//    }
//    if (type2 == VTKElemType::None) {
//      type2 = this_type;
//    }
//    if (type2 == this_type) {
//      continue;
//    }
//    // third type found. Not valid!
//    return MeshType::None;
//  }
//  // Determine the mesh type from the 1 or 2 VTK elem types.
//  if (type1 == VTKElemType::Triangle && type2 == VTKElemType::None) {
//    return MeshType::Tri;
//  }
//  if (type1 == VTKElemType::Quad && type2 == VTKElemType::None) {
//    return MeshType::Quad;
//  }
//  if ((type1 == VTKElemType::Triangle && type2 == VTKElemType::Quad) ||
//      (type2 == VTKElemType::Triangle && type1 == VTKElemType::Quad)) {
//    return MeshType::TriQuad;
//  }
//  if (type1 == VTKElemType::QuadraticTriangle && type2 == VTKElemType::None) {
//    return MeshType::QuadraticTri;
//  }
//  if (type1 == VTKElemType::QuadraticQuad && type2 == VTKElemType::None) {
//    return MeshType::QuadraticQuad;
//  }
//  if ((type1 == VTKElemType::QuadraticTriangle && type2 == VTKElemType::QuadraticQuad) ||
//      (type2 == VTKElemType::QuadraticTriangle && type1 == VTKElemType::QuadraticQuad)) {
//    return MeshType::QuadraticTriQuad;
//  }
//  return MeshType::None;
//}

} // namespace um2
