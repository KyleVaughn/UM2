#pragma once

#include <um2/geometry/point.hpp>
#include <um2/geometry/polygon.hpp>
#include <um2/mesh/element_types.hpp>
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
// A data structure for storing a mesh, or any collection of polytopes, and data
// associated with the polytopes. This data structure can be used to:
// - read/write a mesh and its data from/to a file
// - convert between mesh data structures
// - generate submeshes
// - perform mesh operations without assumptions about manifoldness, etc.
//
// ASSUMES THAT EACH ELEMENT HAS VERTICES IN THE SAME Z-COOORDINATE PLANE.
// Element 1 and element 2 can have different z-coordinates, but all vertices
// of element 1 will have the same z-coordinate, and all vertices of element 2
// will have the same z-coordinate.
//
// Note: due to the generality of the data structure, there is effectively a
// switch statement in every method. This is not ideal for performance.
// See FaceVertexMesh for a more efficient, but less general, data
// structure.

class PolytopeSoup
{

  bool _is_morton_sorted = false;
  Vector<Point3> _vertices;
  Vector<VTKElemType> _element_types;
  Vector<I> _element_offsets; // A prefix sum of the number of vertices in each element
  Vector<I> _element_conn;    // Vertex IDs of each element

  Vector<String> _elset_names;
  Vector<I> _elset_offsets;      // A prefix sum of the number of elements in each elset
  Vector<I> _elset_ids;          // Element IDs of each elset
  Vector<Vector<F>> _elset_data; // Data associated with each elset

  //==============================================================================
  // Private Methods
  //==============================================================================

  void
  writeXDMFUniformGrid(String const & name, Vector<String> const & material_names,
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
                  String const & h5path, Vector<String> const & material_names) const;

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
  addElement(VTKElemType type, Vector<I> const & conn) -> I;

  auto
  addElset(String const & name, Vector<I> const & ids, Vector<F> data = {}) -> I;

  auto
  addVertex(F x, F y, F z = 0) -> I;

  auto
  addVertex(Point3 const & p) -> I;

  PURE [[nodiscard]] auto
  compareTo(PolytopeSoup const & other) const -> int;

  void
  getElement(I i, VTKElemType & type, Vector<I> & conn) const;

  [[nodiscard]] auto
  getElementBoundingBox(I i) const -> AxisAlignedBox3;

  [[nodiscard]] auto
  getElementCentroid(I i) const -> Point3;

  [[nodiscard]] constexpr auto
  getElemTypes() const -> Vector<VTKElemType>;

  void
  getElset(I i, String & name, Vector<I> & ids, Vector<F> & data) const;

  void
  getMaterialIDs(Vector<MaterialID> & material_ids,
                 Vector<String> const & material_names) const;

  void
  getMaterialNames(Vector<String> & material_names) const;

  PURE [[nodiscard]] constexpr auto
  getMeshType() const -> MeshType;

  void
  getSubmesh(String const & elset_name, PolytopeSoup & submesh) const;

  PURE [[nodiscard]] constexpr auto
  getVertex(I i) const -> Point3 const &;

  // Sort the vertices and elements
  void
  mortonSort();

  void
  mortonSortElements();

  void
  mortonSortVertices();

  PURE [[nodiscard]] constexpr auto
  numElems() const -> I;

  PURE [[nodiscard]] constexpr auto
  numElsets() const -> I;

  PURE [[nodiscard]] constexpr auto
  numVerts() const -> I;

  void
  read(String const & filename);

  void
  reserveMoreElements(VTKElemType elem_type, I num_elems);

  void
  reserveMoreVertices(I num_verts);

  void
  sortElsets();

  void
  translate(Point3 const & v);

  void
  write(String const & filename) const;

  void
  writeXDMF(String const & filepath) const;

}; // struct PolytopeSoup

//==============================================================================
// Methods
//==============================================================================

PURE constexpr auto
PolytopeSoup::numVerts() const -> I
{
  return _vertices.size();
}

PURE constexpr auto
PolytopeSoup::numElsets() const -> I
{
  return _elset_names.size();
}

PURE constexpr auto
PolytopeSoup::numElems() const -> I
{
  return _element_types.size();
}

PURE constexpr auto
PolytopeSoup::getVertex(I const i) const -> Point3 const &
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
      el_types.push_back(this_type);
    }
  }
  return el_types;
}

PURE constexpr auto
PolytopeSoup::getMeshType() const -> MeshType
{
  // Loop through the element types to determine which 1 or 2 mesh types are
  // present.
  VTKElemType type1 = VTKElemType::None;
  VTKElemType type2 = VTKElemType::None;
  for (auto const & this_type : _element_types) {
    if (type1 == VTKElemType::None) {
      type1 = this_type;
    }
    if (type1 == this_type) {
      continue;
    }
    if (type2 == VTKElemType::None) {
      type2 = this_type;
    }
    if (type2 == this_type) {
      continue;
    }
    return MeshType::None;
  }
  // Determine the mesh type from the 1 or 2 VTK elem types.
  if (type1 == VTKElemType::Triangle && type2 == VTKElemType::None) {
    return MeshType::Tri;
  }
  if (type1 == VTKElemType::Quad && type2 == VTKElemType::None) {
    return MeshType::Quad;
  }
  if ((type1 == VTKElemType::Triangle && type2 == VTKElemType::Quad) ||
      (type2 == VTKElemType::Triangle && type1 == VTKElemType::Quad)) {
    return MeshType::TriQuad;
  }
  if (type1 == VTKElemType::QuadraticTriangle && type2 == VTKElemType::None) {
    return MeshType::QuadraticTri;
  }
  if (type1 == VTKElemType::QuadraticQuad && type2 == VTKElemType::None) {
    return MeshType::QuadraticQuad;
  }
  if ((type1 == VTKElemType::QuadraticTriangle && type2 == VTKElemType::QuadraticQuad) ||
      (type2 == VTKElemType::QuadraticTriangle && type1 == VTKElemType::QuadraticQuad)) {
    return MeshType::QuadraticTriQuad;
  }
  return MeshType::None;
}

} // namespace um2
