#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

// External dependencies
#if UM2_USE_HDF5
#  include <H5Cpp.h>
#endif

#if UM2_USE_PUGIXML
#  include <pugixml.hpp>
#endif

#define UM2_HAS_XDMF (UM2_USE_HDF5 && UM2_USE_PUGIXML)

namespace um2
{

//==============================================================================
// POLYTOPE SOUP
//==============================================================================
// A data structure for storing a mesh, or any collection of polytopes, and data
// associated with the polytopes. This data structure can be used to:
// - read/write a mesh and its data from/to a file
// - convert between mesh data structures
// - generate subsets

class PolytopeSoup
{

  Vector<Point3F> _vertices;
  Vector<VTKElemType> _element_types;
  Vector<Int> _element_offsets; // A prefix sum of the number of vertices in each element
  Vector<Int> _element_conn;    // Vertex IDs of each element

  Vector<String> _elset_names;
  Vector<Int> _elset_offsets; // A prefix sum of the number of elements in each elset
  Vector<Int> _elset_ids;     // Element IDs of each elset (must be sorted)
  Vector<Vector<Float>> _elset_data; // Data associated with each elset

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr PolytopeSoup() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  PolytopeSoup(String const & filename);

  //==============================================================================
  // Member access
  //==============================================================================

  PURE [[nodiscard]] constexpr auto
  vertices() const -> Vector<Point3F> const &;

  PURE [[nodiscard]] constexpr auto
  elementTypes() const -> Vector<VTKElemType> const &;

  PURE [[nodiscard]] constexpr auto
  elementOffsets() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elementConnectivity() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elsetNames() const -> Vector<String> const &;

  PURE [[nodiscard]] constexpr auto
  elsetOffsets() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elsetIDs() const -> Vector<Int> const &;

  PURE [[nodiscard]] constexpr auto
  elsetData() const -> Vector<Vector<Float>> const &;

  //==============================================================================
  // Capacity
  //==============================================================================

  PURE [[nodiscard]] constexpr auto
  numVertices() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElements() const -> Int;

  PURE [[nodiscard]] constexpr auto
  numElsets() const -> Int;

  //==============================================================================
  // Getters
  //==============================================================================

  PURE [[nodiscard]] inline auto
  getVertex(Int i) const -> Point3F const &;

  void
  getElement(Int i, VTKElemType & type, Vector<Int> & conn) const;

  // The unique element types in the soup
  [[nodiscard]] inline auto
  getElemTypes() const -> Vector<VTKElemType>;

  void
  getElset(Int i, String & name, Vector<Int> & ids, Vector<Float> & data) const;

  void
  getElset(String const & name, Vector<Int> & ids, Vector<Float> & data) const;

  //==============================================================================
  // Modifiers
  //==============================================================================

  auto
  addVertex(Float x, Float y, Float z = 0) -> Int;

  auto
  addVertex(Point3F const & p) -> Int;

  auto
  addElement(VTKElemType type, Vector<Int> const & conn) -> Int;

  auto
  addElset(String const & name, Vector<Int> const & ids, Vector<Float> data = {}) -> Int;

  constexpr void
  translate(Point3F const & v) noexcept;

  void
  reserveMoreElements(VTKElemType elem_type, Int num_elems);

  void
  reserveMoreVertices(Int num_verts);

  void
  sortElsets();

  auto
  operator+=(PolytopeSoup const & other) noexcept -> PolytopeSoup &;

  //==============================================================================
  // Methods
  //==============================================================================

  void
  getSubset(String const & elset_name, PolytopeSoup & subset) const;

  PURE [[nodiscard]] auto
  compare(PolytopeSoup const & other) const -> int;

  //==============================================================================
  // I/O
  //==============================================================================

  void
  read(String const & filename);

  void
  write(String const & filename) const;

}; // class PolytopeSoup

//==============================================================================
// Free functions
//==============================================================================

#if UM2_HAS_XDMF
void
writeXDMFUniformGrid(String const & name, pugi::xml_node & xdomain, H5::H5File & h5file,
                     String const & h5filename, String const & h5path,
                     PolytopeSoup const & soup, Point3F const & origin = {0, 0, 0});

void
readXDMFUniformGrid(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                    String const & h5filename, PolytopeSoup & soup);
#endif

//==============================================================================
// Member access
//==============================================================================

PURE constexpr auto
PolytopeSoup::vertices() const -> Vector<Point3F> const &
{
  return _vertices;
}

PURE constexpr auto
PolytopeSoup::elementTypes() const -> Vector<VTKElemType> const &
{
  return _element_types;
}

PURE constexpr auto
PolytopeSoup::elementOffsets() const -> Vector<Int> const &
{
  return _element_offsets;
}

PURE constexpr auto
PolytopeSoup::elementConnectivity() const -> Vector<Int> const &
{
  return _element_conn;
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

//==============================================================================
// Capacity
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

//==============================================================================
// Getters
//==============================================================================

PURE inline auto
PolytopeSoup::getVertex(Int const i) const -> Point3F const &
{
  ASSERT(i < _vertices.size());
  return _vertices[i];
}

inline auto
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

//==============================================================================
// Modifiers
//==============================================================================

constexpr void
PolytopeSoup::translate(Point3F const & v) noexcept
{
  for (auto & p : _vertices) {
    p += v;
  }
}

} // namespace um2
