#pragma once

#include <um2/common/log.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polygon.hpp>    
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>
#include <um2/stdlib/sto.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

#include <cstring>
#include <charconv>
#include <concepts>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

// External dependencies
#include <H5Cpp.h>
#include <pugixml.hpp>

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

template <std::floating_point T, std::signed_integral I>
class PolytopeSoup {

  bool _is_morton_sorted = false;
  Vector<Point3<T>> _vertices;
  Vector<VTKElemType> _element_types;
  Vector<I> _element_offsets; // A prefix sum of the number of vertices in each element
  Vector<I> _element_conn;    // Vertex IDs of each element

  Vector<String> _elset_names;
  Vector<I> _elset_offsets;      // A prefix sum of the number of elements in each elset
  Vector<I> _elset_ids;          // Element IDs of each elset
  Vector<Vector<T>> _elset_data; // Data associated with each elset

  public:

  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr PolytopeSoup() = default;

  //==============================================================================
  // Methods 
  //==============================================================================

  auto 
  addElement(VTKElemType type, Vector<I> const & conn) -> Size;

  auto
  addElset(String const & name, Vector<I> const & ids, Vector<T> data = {}) -> Size;

  auto
  addVertex(T x, T y, T z = 0) -> Size;

  auto
  addVertex(Point3<T> const & p) -> Size;

  [[nodiscard]] auto
  compareTo(PolytopeSoup const & other) const -> int;

  void
  getElement(Size i, VTKElemType & type, Vector<I> & conn) const; 

  PURE [[nodiscard]] constexpr auto
  getElemTypes() const -> Vector<VTKElemType>;

  void
  getElset(Size i, String & name, Vector<I> & ids, Vector<T> & data) const;

  void
  getMaterialIDs(Vector<MaterialID> & material_ids,
                 Vector<String> const & material_names) const;

  void
  getMaterialNames(Vector<String> & material_names) const;

  PURE [[nodiscard]] constexpr auto
  getMeshType() const -> MeshType;

  void
  getSubmesh(String const & elset_name, PolytopeSoup<T, I> & submesh) const;

  PURE [[nodiscard]] constexpr auto
  getVertex(Size i) const -> Point3<T> const &;

  PURE [[nodiscard]] constexpr auto
  numElems() const -> Size;

  PURE [[nodiscard]] constexpr auto
  numElsets() const -> Size;

  PURE [[nodiscard]] constexpr auto
  numVerts() const -> Size;
  
  void
  read(String const & filename);

  void
  sortElsets();

  void
  write(String const & filename) const;

  //==============================================================================
  // Operators
  //==============================================================================

  auto
  operator==(PolytopeSoup const & other) const -> bool;

  //============================================================================
  // Hidden Methods
  //============================================================================

  HIDDEN void
  writeXDMF(String const & filepath) const;

  HIDDEN void
  writeXDMFUniformGrid(
     String const & name,
     Vector<String> const & material_names,
     pugi::xml_node & xdomain,
     H5::H5File & h5file,
     String const & h5filename,
     String const & h5path) const;

  HIDDEN void
  writeXDMFGeometry(
      pugi::xml_node & xgrid,
      H5::Group & h5group,
      String const & h5filename,
      String const & h5path) const;

  HIDDEN void
  writeXDMFTopology(
      pugi::xml_node & xgrid,
      H5::Group & h5group,
      String const & h5filename,
      String const & h5path) const;

  HIDDEN void
  writeXDMFElsets(
      pugi::xml_node & xgrid,
      H5::Group & h5group,
      String const & h5filename,
      String const & h5path,
      Vector<String> const & material_names) const;

}; // struct PolytopeSoup

//==============================================================================
// Methods 
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::numVerts() const -> Size
{
  return _vertices.size();
}

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::numElsets() const -> Size
{
  return _elset_names.size();
}

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::numElems() const -> Size
{
  return _element_types.size();
}

template <std::floating_point T, std::signed_integral I>
void 
PolytopeSoup<T, I>::getElement(Size const i, VTKElemType & type, Vector<I> & conn) const
{
  ASSERT(i < _element_types.size());
  type = _element_types[i];
  Size const istart = static_cast<Size>(_element_offsets[i]); 
  Size const iend = static_cast<Size>(_element_offsets[i + 1]);
  auto const n = iend - istart; 
  if (n != conn.size()) {
    conn.resize(n);
  }
  for (Size j = 0; j < n; ++j) {
    conn[j] = _element_conn[istart + j];
  }
}

template <std::floating_point T, std::signed_integral I>
void 
PolytopeSoup<T, I>::getElset(Size const i, String & name, Vector<I> & ids, Vector<T> & data) const
{
  ASSERT(i < _elset_names.size());
  name = _elset_names[i];
  Size const istart = static_cast<Size>(_elset_offsets[i]);
  Size const iend = static_cast<Size>(_elset_offsets[i + 1]);
  auto const n = iend - istart;
  if (n != ids.size()) {
    ids.resize(n);
  }
  for (Size j = 0; j < n; ++j) {
    ids[j] = _elset_ids[istart + j];
  }
  if (!_elset_data[i].empty()) {
    if (data.size() != n) {
      data.resize(n);
    }
    for (Size j = 0; j < n; ++j) {
      data[j] = _elset_data[i][istart + j];
    }
  }
}

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::getVertex(Size const i) const -> Point3<T> const &
{
  ASSERT(i < _vertices.size());
  return _vertices[i];
}

//==============================================================================
// Operators
//==============================================================================

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::operator==(PolytopeSoup const & other) const -> bool
{
  return compareTo(other) == 0;
}

//==============================================================================
// addVertex
//==============================================================================

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::addVertex(T x, T y, T z) -> Size
{
  _vertices.emplace_back(x, y, z);
  return _vertices.size() - 1;
}

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::addVertex(Point3<T> const & p) -> Size
{
  _vertices.push_back(p);
  return _vertices.size() - 1;
}

//==============================================================================
// addElement
//==============================================================================

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::addElement(VTKElemType const type, Vector<I> const & conn) -> Size
{
  _element_types.push_back(type);
  if (_element_offsets.empty()) {
    _element_offsets.push_back(0);
  }
  _element_offsets.push_back(_element_offsets.back() + static_cast<I>(conn.size()));
  for (auto const & id : conn) {
    ASSERT(id < _vertices.size());
    // cppcheck-suppress useStlAlgorithm;
    _element_conn.push_back(id);
  }
  return _element_types.size() - 1;
}

//==============================================================================
// addElset
//==============================================================================

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::addElset(String const & name, Vector<I> const & ids, Vector<T> data) -> Size
{
  LOG_DEBUG("Adding elset: " + name);

  for (auto const & this_name : _elset_names) {
    // cppcheck-suppress useStlAlgorithm; justification: This is more clear.
    if (this_name == name) {
      LOG_ERROR("Elset " + name + " already exists.");
      return -1;
    }
  }

  Size const num_ids = ids.size();
  if (num_ids == 0) {
    LOG_ERROR("Elset ids" + name + " is empty.");
    return -1;
  }

  if (!data.empty() && (data.size() != num_ids)) {
    LOG_ERROR("Elset data size does not match the number of ids.");
    return -1;
  }

  _elset_names.emplace_back(name);
  if (_elset_offsets.empty()) {
    _elset_offsets.push_back(0);
  }

  Size const old_num_ids = _elset_ids.size();
  Size const new_num_ids = old_num_ids + num_ids;
  _elset_offsets.push_back(static_cast<I>(new_num_ids));
  _elset_ids.resize(new_num_ids);
  um2::copy(ids.begin(), ids.end(), _elset_ids.data() + old_num_ids);
#ifndef NDEBUG
  for (auto const & id : _elset_ids) {
    ASSERT(id < _element_types.size());
  }
#endif
  _elset_data.emplace_back(um2::move(data));
  return _elset_names.size() - 1;
}

//==============================================================================
// compareTo
//==============================================================================

template <std::floating_point T, std::signed_integral I>
auto
PolytopeSoup<T, I>::compareTo(PolytopeSoup<T, I> const & other) const -> int
{

  if (_is_morton_sorted != other._is_morton_sorted) {
    return 1;
  }
  if (_vertices.size() != other._vertices.size()) {
    return 2;
  }
  auto const compare = [](Point3<T> const & a, Point3<T> const & b) -> bool {
    return um2::isApprox(a, b);
  };
  if (!std::equal(_vertices.cbegin(), _vertices.cend(), other._vertices.cbegin(),
                  compare)) {
    return 3;
  }
  if (_element_types.size() != other._element_types.size()) {
    return 4;
  }
  if (!std::equal(_element_types.cbegin(), _element_types.cend(),
                  other._element_types.cbegin())) {
    return 5;
  }
  if (_element_offsets.size() != other._element_offsets.size()) {
    return 6;
  }
  if (!std::equal(_element_offsets.cbegin(), _element_offsets.cend(),
                  other._element_offsets.cbegin())) {
    return 7;
  }
  if (_element_conn.size() != other._element_conn.size()) {
    return 8;
  }
  if (!std::equal(_element_conn.cbegin(), _element_conn.cend(),
                  other._element_conn.cbegin())) {
    return 9;
  }
  if (_elset_names.size() != other._elset_names.size()) {
    return 10;
  }
  if (!std::equal(_elset_names.cbegin(), _elset_names.cend(),
                  other._elset_names.cbegin())) {
    return 11;
  }
  if (_elset_offsets.size() != other._elset_offsets.size()) {
    return 12;
  }
  if (!std::equal(_elset_offsets.cbegin(), _elset_offsets.cend(),
                  other._elset_offsets.cbegin())) {
    return 13;
  }
  if (_elset_ids.size() != other._elset_ids.size()) {
    return 14;
  }
  if (!std::equal(_elset_ids.cbegin(), _elset_ids.cend(),
                  other._elset_ids.cbegin())) {
    return 15;
  }
  if (_elset_data.size() != other._elset_data.size()) {
    return 16;
  }
  if (!std::equal(_elset_data.cbegin(), _elset_data.cend(),
                  other._elset_data.cbegin())) {
    return 17;
  }
  return 0;
}

//==============================================================================
// getElemTypes
//==============================================================================
// Get all the element types in the mesh.

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::getElemTypes() const -> Vector<VTKElemType>
{
  Vector<VTKElemType> el_types;
  for (auto const & this_type : _element_types) {
    bool found = false;
    for (auto const & that_type : el_types) {
      // cppcheck-suppress useStlAlgorithm
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

//==============================================================================
// getMeshType
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::getMeshType() const -> MeshType
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

//==============================================================================
// sortElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
PolytopeSoup<T, I>::sortElsets()
{
  using NameIndexPair = std::pair<String, I>;
  Size const num_elsets = _elset_names.size();
  Vector<NameIndexPair> elset_name_index_pairs(num_elsets);
  for (Size i = 0; i < num_elsets; ++i) {
    elset_name_index_pairs[i] = std::make_pair(_elset_names[i], static_cast<I>(i));
  }
  // Sort the vector by the elset names.
  std::sort(elset_name_index_pairs.begin(), elset_name_index_pairs.end(),
            [](NameIndexPair const & a, NameIndexPair const & b) -> bool {
              return a.first < b.first;
            });
  // Create a vector to store the sorted elset ids.
  Vector<I> elset_offsets(_elset_offsets.size());
  Vector<I> elset_ids(_elset_ids.size());
  Vector<Vector<T>> elset_data(_elset_data.size());
  I offset = 0;
  for (Size i = 0; i < num_elsets; ++i) {
    auto const & name = elset_name_index_pairs[i].first;
    auto const & index = elset_name_index_pairs[i].second;
    auto const iold = static_cast<Size>(index);
    _elset_names[i] = name;
    I const len = _elset_offsets[iold + 1] - _elset_offsets[iold];
    elset_offsets[i] = offset;
    elset_offsets[i + 1] = offset + len;
    copy(_elset_ids.begin() + _elset_offsets[iold],
         _elset_ids.begin() + _elset_offsets[iold + 1],
         elset_ids.begin() + elset_offsets[i]);
    elset_data[i] = um2::move(_elset_data[iold]);
    offset += len;
  }
  _elset_offsets = um2::move(elset_offsets);
  _elset_ids = um2::move(elset_ids);
  _elset_data = um2::move(elset_data);
}

//==============================================================================
// getMaterialNames
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
PolytopeSoup<T, I>::getMaterialNames(Vector<String> & material_names) const
{
  material_names.clear();
  String const mat_prefix = "Material_";
  for (auto const & elset_name : _elset_names) {
    if (elset_name.starts_with(mat_prefix)) {
      // cppcheck-suppress useStlAlgorithm; justification: Different behavior.
      material_names.emplace_back(elset_name);
    }
  }
}

//==============================================================================
// getSubmesh
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
PolytopeSoup<T, I>::getSubmesh(String const & elset_name,
                               PolytopeSoup<T, I> & submesh) const
{
  LOG_DEBUG("Extracting submesh for elset: " + elset_name);

  // Find the elset with the given name.
  Size elset_index = 0;
  bool found = false;
  for (Size i = 0; i < _elset_names.size(); ++i) {
    if (_elset_names[i] == elset_name) {
      elset_index = i;
      found = true;
    }
  }
  if (!found) {
    Log::error("getSubmesh: Elset '" + elset_name + "' not found");
    return;
  }

  auto const submesh_elset_start = static_cast<Size>(_elset_offsets[elset_index]);
  auto const submesh_elset_end = static_cast<Size>(_elset_offsets[elset_index + 1]);
  auto const submesh_num_elements = submesh_elset_end - submesh_elset_start;
  Vector<I> element_ids(submesh_num_elements);
  for (Size i = 0; i < submesh_num_elements; ++i) {
    element_ids[i] = _elset_ids[submesh_elset_start + i];
  }
  ASSERT(um2::is_sorted(element_ids.cbegin(), element_ids.cend()));

  // Get the unique vertices in all the elements.
  Vector<I> vertex_ids;
  vertex_ids.reserve(submesh_num_elements * 3);
  for (Size i = 0; i < submesh_num_elements; ++i) {
    auto const element_id = static_cast<Size>(element_ids[i]);
    auto const element_start = static_cast<Size>(_element_offsets[element_id]);
    auto const element_end = static_cast<Size>(_element_offsets[element_id + 1]);
    auto const element_len = element_end - element_start;
    for (Size j = 0; j < element_len; ++j) {
      I const vertex_id = _element_conn[element_start + j];
      vertex_ids.push_back(vertex_id);
    }
  }
  std::sort(vertex_ids.begin(), vertex_ids.end());
  Vector<I> unique_vertex_ids = vertex_ids; 
  auto * const last = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
  auto const num_unique_verts = static_cast<Size>(last - unique_vertex_ids.cbegin());
  
  // Add each of the vertices to the submesh.
  for (Size i = 0; i < num_unique_verts; ++i) {
    auto const vertex_id = static_cast<Size>(unique_vertex_ids[i]);
    auto const & vertex = _vertices[vertex_id];
    submesh.addVertex(vertex);
  }

  // For each element, add it to the submesh, remapping the vertex IDs
  // unique_vertex_ids[i] is the old vertex id, and i is the new vertex id.
  Size conn_len = 0;
  Vector<I> conn;
  for (Size i = 0; i < submesh_num_elements; ++i) {
    auto const element_id = static_cast<Size>(element_ids[i]);
    auto const element_type = _element_types[element_id];
    auto const element_start = static_cast<Size>(_element_offsets[element_id]);
    auto const element_end = static_cast<Size>(_element_offsets[element_id + 1]);
    auto const element_len = element_end - element_start;
    if (element_len != conn_len) {
      conn.resize(element_len);
      conn_len = element_len;
    }
    for (Size j = 0; j < element_len; ++j) {
      I const old_vertex_id = _element_conn[element_start + j];
      auto * const it = std::lower_bound(unique_vertex_ids.begin(), last, old_vertex_id);
      auto const new_vertex_id = static_cast<I>(it - unique_vertex_ids.cbegin());
      ASSERT(*it == old_vertex_id);
      conn[j] = new_vertex_id;
    }
    submesh.addElement(element_type, conn);
  }

  // If the intersection of this elset and another elset is non-empty, then we need to
  // add the itersection as an elset and remap the elset IDs using the element_ids
  // vector.
  //
  // element_ids[i] is the old element id, and i is the new element id.
  Size const num_elsets = _elset_names.size();
  for (Size i = 0; i < num_elsets; ++i) {
    if (i == elset_index) {
      continue;
    }
    auto const elset_start = static_cast<Size>(_elset_offsets[i]);
    auto const elset_end = static_cast<Size>(_elset_offsets[i + 1]);
    auto * const elset_ids_begin = addressof(_elset_ids[elset_start]);
    auto * const elset_ids_end = elset_ids_begin + (elset_end - elset_start);
    std::vector<I> intersection;
    std::set_intersection(element_ids.begin(), element_ids.end(), elset_ids_begin,
                          elset_ids_end, std::back_inserter(intersection));
    if (intersection.empty()) {
      continue;
    }
    auto const & name = _elset_names[i];
    auto const num_ids = static_cast<Size>(intersection.size());
    Vector<I> ids(num_ids);
    // Remap the element IDs
    for (Size j = 0; j < num_ids; ++j) {
      I const old_element_id = intersection[static_cast<size_t>(j)];
      auto * const it = std::lower_bound(element_ids.cbegin(), element_ids.cend(), old_element_id);
      auto const new_element_id = static_cast<I>(it - element_ids.cbegin());
      ASSERT(*it == old_element_id);
      ids[j] = new_element_id;
    }
    if (_elset_data[i].empty()) {
      submesh.addElset(name, ids);
      continue;
    } 
    // There is data
    Vector<T> elset_data(num_ids);
    Vector<T> const & this_elset_data = _elset_data[i];
    for (Size j = 0; j < num_ids; ++j) {
      I const old_element_id = intersection[static_cast<size_t>(j)];
      auto * const it = std::lower_bound(elset_ids_begin, elset_ids_end, old_element_id);
      ASSERT(*it == old_element_id);
      auto const idx = static_cast<Size>(it - elset_ids_begin);
      elset_data[j] = this_elset_data[idx];
    }
    submesh.addElset(name, ids, elset_data);
  }
}

//==============================================================================
// getMaterialIDs
//==============================================================================

 template <std::floating_point T, std::signed_integral I>
void
 PolytopeSoup<T, I>::getMaterialIDs(Vector<MaterialID> & material_ids,
                                    Vector<String> const & material_names) const
{
  material_ids.resize(numElems());
  um2::fill(material_ids.begin(), material_ids.end(), static_cast<MaterialID>(-1));
  Size const nmats = material_names.size();
  for (Size i = 0; i < nmats; ++i) {
    String const & mat_name = material_names[i];
    for (Size j = 0; j < _elset_names.size(); ++j) {
      if (_elset_names[j] == mat_name) {
        auto const start = static_cast<Size>(_elset_offsets[j]);
        auto const end = static_cast<Size>(_elset_offsets[j + 1]);
        for (Size k = start; k < end; ++k) {
          auto const elem = static_cast<Size>(_elset_ids[k]);
          if (material_ids[elem] != -1) {
            Log::error("Element " + toString(elem) + " has multiple materials");
          }
          material_ids[elem] = static_cast<MaterialID>(i);
        } // for k
        break;
      } // if elset_names[j] == mat_name
    }   // for j
  }     // for i
  if (std::any_of(material_ids.cbegin(), material_ids.cend(),
                  [](MaterialID const mat_id) { return mat_id == -1; })) {
    Log::error("Some elements have no material");
  }
}

//==============================================================================-
// IO for ABAQUS files.
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
abaqusParseNodes(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
{
  // Would love to use chars_format here, but it bugs out on "0.5" occasionally
  LOG_DEBUG("Parsing nodes");
  while (std::getline(file, line) && line[0] != '*') {
    // Format: node_id, x, y, z
    // Skip ID
    size_t last = line.find(',', 0);
    size_t next = line.find(',', last + 2);
    // Read coordinates
    T const x = sto<T>(line.substr(last + 2, next - last - 2));
    last = next;
    next = line.find(',', last + 2);
    T const y = sto<T>(line.substr(last + 2, next - last - 2));
    T const z = sto<T>(line.substr(next + 2));
    soup.addVertex(x, y, z);
  }
} // abaqusParseNodes

template <std::floating_point T, std::signed_integral I>
static void
abaqusParseElements(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
{
  LOG_DEBUG("Parsing elements");
  //  "*ELEMENT, type=CPS".size() = 18
  //  CPS3 is a 3-node triangle
  //  CPS4 is a 4-node quadrilateral
  //  CPS6 is a 6-node quadratic triangle
  //  CPS8 is a 8-node quadratic quadrilateral
  //  Hence, line[18] is the offset of the element type
  //  ASCII code for '0' is 48, so line[18] - 48 is the offset
  //  as an integer
  //
  if (line[15] != 'C' || line[16] != 'P' || line[17] != 'S') {
    LOG_ERROR("Only CPS elements are supported");
    return;
  }
  I const offset = static_cast<I>(line[18]) - 48;
  VTKElemType this_type = VTKElemType::Vertex;
  switch (offset) {
  case 3:
    this_type = VTKElemType::Triangle;
    break;
  case 4:
    this_type = VTKElemType::Quad;
    break;
  case 6:
    this_type = VTKElemType::QuadraticTriangle;
    break;
  case 8:
    this_type = VTKElemType::QuadraticQuad;
    break;
  default: {
    LOG_ERROR("AbaqusCellType CPS" + toString(offset) + " is not supported");
    break;
  }
  }
  Size const verts_per_elem = verticesPerElem(this_type);
  Vector<I> conn(verticesPerElem(this_type));
  while (std::getline(file, line) && line[0] != '*') {
    LOG_TRACE("Line: " + String(line.c_str()));
    std::string_view const line_view = line;
    // For each element, read the element ID and the node IDs
    // Format: id, n1, n2, n3, n4, n5 ...
    // Skip ID
    size_t last = line_view.find(',', 0);
    size_t next = line_view.find(',', last + 2);
    I id = -1;
    // Read the first N-1 node IDs
    for (Size i = 0; i < verts_per_elem - 1; ++i) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      LOG_TRACE("Node ID: " + toString(id));
      ASSERT(id > 0);
      conn[i] = id - 1; // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 2);
    }
    // Read last node ID
    std::from_chars(line_view.data() + last + 2, line_view.data() + line_view.size(), id);
    ASSERT(id > 0);
    conn[verts_per_elem - 1] = id - 1; // ABAQUS is 1-indexed
    soup.addElement(this_type, conn);
  }
} // abaqusParseElements

template <std::floating_point T, std::signed_integral I>
static void
abaqusParseElsets(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
{
  LOG_DEBUG("Parsing elsets");
  std::string_view line_view = line;
  // "*ELSET,ELSET=".size() = 13
  std::string const elset_name_std{line_view.substr(13, line_view.size() - 13)};
  String const elset_name(elset_name_std.c_str());
  Vector<I> elset_ids;
  while (std::getline(file, line) && line[0] != '*') {
    line_view = line;
    // Add each element ID to the elset
    // Format: id, id, id, id, id,
    // Note, line ends in ", " or ","
    // First ID
    size_t last = 0;
    size_t next = line_view.find(',');
    I id;
    std::from_chars(line_view.data(), line_view.data() + next, id);
    ASSERT(id > 0);
    elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
    last = next;
    next = line_view.find(',', last + 1);
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      ASSERT(id > 0);
      elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 1);
    }
  }
  ASSERT(um2::is_sorted(elset_ids.cbegin(), elset_ids.cend()));
  soup.addElset(elset_name, elset_ids);
} // abaqusParseElsets

template <std::floating_point T, std::signed_integral I>
void
readAbaqusFile(String const & filename, PolytopeSoup<T, I> & soup)
{
  LOG_INFO("Reading Abaqus file: " + filename);

  // Open file
  std::ifstream file(filename.c_str());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: " + filename);
    return;
  }

  // Read file
  std::string line;
  bool loop_again = false;
  while (loop_again || std::getline(file, line)) {
    loop_again = false;
    if (line.starts_with("*NODE")) {
      abaqusParseNodes<T>(soup, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELEMENT")) {
      abaqusParseElements<T, I>(soup, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELSET")) {
      abaqusParseElsets<T, I>(soup, line, file);
      loop_again = true;
    }
  }
  soup.sortElsets();
  file.close();
  LOG_INFO("Finished reading Abaqus file: " + filename);
} // readAbaqusFile

//==============================================================================
// IO for XDMF files
//==============================================================================

// Turn off useless cast warnings, since the casts are only useless for certain
// CMake configurations.
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

template <typename T>
static inline auto
getH5DataType() -> H5::PredType
{
  // NOLINTNEXTLINE(bugprone-branch-clone) 
  if constexpr (std::same_as<T, float>) {
    return H5::PredType::NATIVE_FLOAT;
  } else if constexpr (std::same_as<T, double>) {
    return H5::PredType::NATIVE_DOUBLE;
  } else if constexpr (std::same_as<T, int8_t>) {
    return H5::PredType::NATIVE_INT8;
  } else if constexpr (std::same_as<T, int16_t>) {
    return H5::PredType::NATIVE_INT16;
  } else if constexpr (std::same_as<T, int32_t>) {
    return H5::PredType::NATIVE_INT32;
  } else if constexpr (std::same_as<T, int64_t>) {
    return H5::PredType::NATIVE_INT64;
  } else if constexpr (std::same_as<T, uint8_t>) {
    return H5::PredType::NATIVE_UINT8;
  } else if constexpr (std::same_as<T, uint16_t>) {
    return H5::PredType::NATIVE_UINT16;
  } else if constexpr (std::same_as<T, uint32_t>) {
    return H5::PredType::NATIVE_UINT32;
  } else if constexpr (std::same_as<T, uint64_t>) {
    return H5::PredType::NATIVE_UINT64;
  } else {
    static_assert(always_false<T>, "Unsupported type");
    return H5::PredType::NATIVE_FLOAT;
  }
}

template <std::floating_point T, std::signed_integral I>
HIDDEN void
PolytopeSoup<T, I>::writeXDMFGeometry(
    pugi::xml_node & xgrid,
    H5::Group & h5group,
    String const & h5filename,
    String const & h5path) const

{
  LOG_DEBUG("Writing XDMF geometry");
  Size const num_verts = _vertices.size();
  bool const is_3d =
      std::any_of(_vertices.cbegin(), _vertices.cend(),
          [](auto const & v) {
              return um2::abs(v[2]) > eps_distance<T>;
          });
  Size const dim = is_3d ? 3 : 2;
  // Create XDMF Geometry node
  auto xgeom = xgrid.append_child("Geometry");
  if (dim == 3) {
    xgeom.append_attribute("GeometryType") = "XYZ";
  } else { // (dim == 2)
    xgeom.append_attribute("GeometryType") = "XY";
  }

  // Create XDMF DataItem node
  auto xdata = xgeom.append_child("DataItem");
  xdata.append_attribute("DataType") = "Float";
  xdata.append_attribute("Dimensions") =
      (toString(num_verts) + " " + toString(dim)).c_str();
  xdata.append_attribute("Precision") = sizeof(T);
  xdata.append_attribute("Format") = "HDF";
  String const h5geompath = h5filename + ":" + h5path + "/Geometry";
  xdata.append_child(pugi::node_pcdata).set_value(h5geompath.c_str());

  // Create HDF5 data space
  hsize_t dims[2] = {static_cast<hsize_t>(num_verts), static_cast<hsize_t>(dim)};
  H5::DataSpace const h5space(2, dims);
  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<T>();
  // Create HDF5 data set
  H5::DataSet const h5dataset = h5group.createDataSet("Geometry", h5type, h5space);
  // Create an xy or xyz array
  Vector<T> xyz(num_verts * dim);
  if (dim == 2) {
    for (Size i = 0; i < num_verts; ++i) {
      xyz[2 * i] = _vertices[i][0];
      xyz[2 * i + 1] = _vertices[i][1];
    }
  } else { // dim == 3
    for (Size i = 0; i < num_verts; ++i) {
      xyz[3 * i] = _vertices[i][0];
      xyz[3 * i + 1] = _vertices[i][1];
      xyz[3 * i + 2] = _vertices[i][2];
    }
  }
  // Write HDF5 data set
  h5dataset.write(xyz.data(), h5type, h5space);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
} // writeXDMFgeometry

template <std::floating_point T, std::signed_integral I>
HIDDEN void
PolytopeSoup<T, I>::writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group,
                  String const & h5filename, String const & h5path) const
{
  LOG_DEBUG("Writing XDMF topology");
  // Create XDMF Topology node
  auto xtopo = xgrid.append_child("Topology");
  Size const nelems = numElems();

  Vector<I> topology;
  String topology_type;
  String dimensions;
  Size nverts = 0;
  auto const elem_type = getElemTypes();
  bool ishomogeneous = true;
  if (elem_type.size() == 1) {
    switch (elem_type[0]) {
    case VTKElemType::Triangle:
      topology_type = "Triangle";
      nverts = 3;
      break;
    case VTKElemType::Quad:
      topology_type = "Quadrilateral";
      nverts = 4;
      break;
    case VTKElemType::QuadraticEdge:
      topology_type = "Edge_3";
      nverts = 3;
      break;
    case VTKElemType::QuadraticTriangle:
      topology_type = "Triangle_6";
      nverts = 6;
      break;
    case VTKElemType::QuadraticQuad:
      topology_type = "Quadrilateral_8";
      nverts = 8;
      break;
    default:
      Log::error("Unsupported polytope type");
    }
    dimensions = toString(nelems) + " " + toString(nverts);
  } else {
    topology_type = "Mixed";
    ishomogeneous = false;
    dimensions = toString(nelems + _element_conn.size());
    topology.resize(nelems + _element_conn.size());
    // Create the topology array (type id + node ids)
    Size topo_ctr = 0;
    for (Size i = 0; i < nelems; ++i) {
      auto const topo_type = static_cast<int8_t>(vtkToXDMFElemType(_element_types[i]));
      if (topo_type == -1) {
        Log::error("Unsupported polytope type");
      }
      topology[topo_ctr] = static_cast<I>(static_cast<unsigned int>(topo_type));
      auto const offset = static_cast<Size>(_element_offsets[i]);
      auto const npts =
          static_cast<Size>(_element_offsets[i + 1] - _element_offsets[i]);
      for (Size j = 0; j < npts; ++j) {
        topology[topo_ctr + j + 1] = _element_conn[offset + j];
      }
      topo_ctr += npts + 1;
    }
  }
  xtopo.append_attribute("TopologyType") = topology_type.c_str();
  xtopo.append_attribute("NumberOfElements") = nelems;
  // Create XDMF DataItem node
  auto xdata = xtopo.append_child("DataItem");
  xdata.append_attribute("DataType") = "Int";
  xdata.append_attribute("Dimensions") = dimensions.c_str();
  xdata.append_attribute("Precision") = sizeof(I);
  xdata.append_attribute("Format") = "HDF";
  String const h5topopath = h5filename + ":" + h5path + "/Topology";
  xdata.append_child(pugi::node_pcdata).set_value(h5topopath.c_str());

  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<I>();
  if (ishomogeneous) {
    // Create HDF5 data space
    hsize_t dims[2] = {static_cast<hsize_t>(nelems), static_cast<hsize_t>(nverts)};
    H5::DataSpace const h5space(2, dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(_element_conn.data(), h5type, h5space);
  } else {
    // Create HDF5 data space
    auto const dims = static_cast<hsize_t>(topology.size());
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(topology.data(), h5type, h5space);
  }
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
} // writeXDMFTopology

template <std::floating_point T, std::signed_integral I>
HIDDEN void
PolytopeSoup<T, I>::writeXDMFElsets(
                            pugi::xml_node & xgrid, H5::Group & h5group,
                            String const & h5filename, String const & h5path,
                            Vector<String> const & material_names) const
{
  LOG_DEBUG("Writing XDMF elsets");
  for (Size i = 0; i < _elset_names.size(); ++i) {
    String const name = _elset_names[i];
    auto const start = static_cast<Size>(_elset_offsets[i]);
    auto const end = static_cast<Size>(_elset_offsets[i + 1]);
    // Create HDF5 data space
    auto dims = static_cast<hsize_t>(end - start);
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data type
    H5::DataType const h5type = getH5DataType<I>();
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet(name.c_str(), h5type, h5space);
    // Write HDF5 data set.
    h5dataset.write(&_elset_ids[start], h5type, h5space);

    // Create XDMF Elset node
    auto xelset = xgrid.append_child("Set");
    xelset.append_attribute("Name") = name.c_str();
    xelset.append_attribute("SetType") = "Cell";
    // Create XDMF DataItem node
    auto xdata = xelset.append_child("DataItem");
    xdata.append_attribute("DataType") = "Int";
    xdata.append_attribute("Dimensions") = end - start;
    xdata.append_attribute("Precision") = sizeof(I);
    xdata.append_attribute("Format") = "HDF";
    String h5elsetpath = h5filename;
    h5elsetpath += ':';
    h5elsetpath += h5path;
    h5elsetpath += '/';
    h5elsetpath += name;
    xdata.append_child(pugi::node_pcdata).set_value(h5elsetpath.c_str());

    if (!_elset_data[i].empty()) {
      if (_elset_names[i].starts_with("Material_")) {
        Log::error("Material elsets should not have data");
      }
      // Create HDF5 data space
      auto const dims_data = static_cast<hsize_t>(_elset_data[i].size());
      H5::DataSpace const h5space_data(1, &dims_data);
      // Create HDF5 data type
      H5::DataType const h5type_data = getH5DataType<T>();
      // Create HDF5 data set
      H5::DataSet const h5dataset_data =
          h5group.createDataSet((name + "_data").c_str(), h5type_data, h5space_data);
      // Write HDF5 data set
      h5dataset_data.write(_elset_data[i].data(), h5type_data, h5space_data);

      // Create XDMF data node
      auto xatt = xelset.append_child("Attribute");
      xatt.append_attribute("Name") = (name + "_data").c_str();
      xatt.append_attribute("Center") = "Cell";
      // Create XDMF DataItem node
      auto xdata2 = xatt.append_child("DataItem");
      xdata2.append_attribute("DataType") = "Float";
      xdata2.append_attribute("Dimensions") = _elset_data[i].size();
      xdata2.append_attribute("Precision") = sizeof(T);
      xdata2.append_attribute("Format") = "HDF";

      String const h5elsetdatapath = h5elsetpath + "_data";
      xdata2.append_child(pugi::node_pcdata).set_value(h5elsetdatapath.c_str());
    }

    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    if (name.starts_with("Material_")) {
      // Get the index of the material in the material_names vector
      Size index = -1;
      for (Size j = 0; j < material_names.size(); ++j) {
        if (name == material_names[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        Log::error("Could not find material name in material_names vector");
      }

      // Create HDF5 data space
      auto const dims_data = static_cast<hsize_t>(end - start);
      H5::DataSpace const h5space_data(1, &dims_data);
      // Create HDF5 data type
      H5::DataType const h5type_data = getH5DataType<int>();
      // Create HDF5 data set
      H5::DataSet const h5dataset_data =
          h5group.createDataSet((name + "_data").c_str(), h5type_data, h5space_data);
      Vector<int> material_ids(end - start, index);
      // Write HDF5 data set
      h5dataset_data.write(material_ids.data(), h5type_data, h5space_data);

      // Create XDMF data node
      auto xatt = xelset.append_child("Attribute");
      xatt.append_attribute("Name") = "Material";
      xatt.append_attribute("Center") = "Cell";
      // Create XDMF DataItem node
      auto xdata2 = xatt.append_child("DataItem");
      xdata2.append_attribute("DataType") = "Int";
      xdata2.append_attribute("Dimensions") = material_ids.size();
      xdata2.append_attribute("Precision") = sizeof(int);
      xdata2.append_attribute("Format") = "HDF";

      String const h5elsetdatapath = h5elsetpath + "_data";
      xdata2.append_child(pugi::node_pcdata).set_value(h5elsetdatapath.c_str());
    }
  }
} // writeXDMFelsets

template <std::floating_point T, std::signed_integral I>
HIDDEN void
PolytopeSoup<T, I>::writeXDMFUniformGrid(
                     String const & name,
                     Vector<String> const & material_names,
                     pugi::xml_node & xdomain,
                     H5::H5File & h5file,
                     String const & h5filename,
                     String const & h5path) const
{
  LOG_DEBUG("Writing XDMF uniform grid");

  // Grid
  pugi::xml_node xgrid = xdomain.append_child("Grid");
  xgrid.append_attribute("Name") = name.c_str();
  xgrid.append_attribute("GridType") = "Uniform";

  // h5
  String const h5grouppath = h5path + "/" + name;
  H5::Group h5group = h5file.createGroup(h5grouppath.c_str());

  writeXDMFGeometry(xgrid, h5group, h5filename, h5grouppath);
  writeXDMFTopology(xgrid, h5group, h5filename, h5grouppath);
  writeXDMFElsets(xgrid, h5group, h5filename, h5grouppath, material_names);
} // writeXDMFUniformGrid

template <std::floating_point T, std::signed_integral I>
HIDDEN void
PolytopeSoup<T, I>::writeXDMF(String const & filepath) const
{
  Log::info("Writing XDMF file: " + filepath);

  // Setup HDF5 file
  // Get the h5 file name
  Size const h5filepath_end = filepath.find_last_of('/') + 1;
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
  String const h5filepath = filepath.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File h5file((h5filepath + h5filename).c_str(), H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Get the material names from elset names, in alphabetical order.
  Vector<String> material_names;
  getMaterialNames(material_names);
  std::sort(material_names.begin(), material_names.end());

  // If there are any materials, add an information node listing them
  if (!material_names.empty()) {
    pugi::xml_node xinfo = xdomain.append_child("Information");
    xinfo.append_attribute("Name") = "Materials";
    String materials;
    for (Size i = 0; i < material_names.size(); ++i) {
      auto const & mat_name = material_names[i];
      String const short_name = mat_name.substr(9, mat_name.size() - 9);
      materials += short_name;
      if (i + 1 < material_names.size()) {
        materials += ", ";
      }
    }
    xinfo.append_child(pugi::node_pcdata).set_value(materials.c_str());
  }

  // Add a uniform grid
  String const h5path;
  String const name = "UM2_mesh";
  writeXDMFUniformGrid(name, material_names, xdomain, h5file, h5filename, h5path);

  // Write the XML file
  xdoc.save_file(filepath.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();
} // writeXDMF
    
template <std::floating_point T, std::signed_integral I, std::floating_point V>
static void
addNodesToMesh(PolytopeSoup<T, I> & mesh, Size const num_verts, Size const num_dimensions,
               H5::DataSet const & dataset, H5::FloatType const & datatype,
               bool const xyz)
{
  Vector<V> data_vec(num_verts * num_dimensions);
  dataset.read(data_vec.data(), datatype);
  // Add the nodes to the mesh
  if (xyz) {
    for (Size i = 0; i < num_verts; ++i) {
      T const x = static_cast<T>(data_vec[i * 3]);
      T const y = static_cast<T>(data_vec[i * 3 + 1]);  
      T const z = static_cast<T>(data_vec[i * 3 + 2]);
      mesh.addVertex(x, y, z);
    }
  } else { // XY
    for (Size i = 0; i < num_verts; ++i) {            
      T const x = static_cast<T>(data_vec[i * 2]);
      T const y = static_cast<T>(data_vec[i * 2 + 1]);
      mesh.addVertex(x, y);
    }
  }
} // addNodesToMesh

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void readXDMFGeometry(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                             String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF geometry");
  pugi::xml_node const xgeometry = xgrid.child("Geometry");
  if (strcmp(xgeometry.name(), "Geometry") != 0) {
    Log::error("XDMF geometry node not found");
    return;
  }
  // Get the geometry type
  String const geometry_type(xgeometry.attribute("GeometryType").value());
  if (geometry_type != "XYZ" && geometry_type != "XY") {
    Log::error("XDMF geometry type not supported: " + geometry_type);
    return;
  }
  // Get the DataItem node
  pugi::xml_node const xdataitem = xgeometry.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    Log::error("XDMF geometry DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Float") {
    Log::error("XDMF geometry data type not supported: " + data_type);
    return;
  }
  // Get the precision
  std::string const precision(xdataitem.attribute("Precision").value());
  if (precision != "4" && precision != "8") {
    Log::error("XDMF geometry precision not supported: " + String(precision.c_str()));
    return;
  }
  // Get the dimensions
  std::string const dimensions(xdataitem.attribute("Dimensions").value());
  size_t const split = dimensions.find_last_of(' ');
  Size const num_verts = sto<Size>(dimensions.substr(0, split));
  Size const num_dimensions = sto<Size>(dimensions.substr(split + 1));
  if (geometry_type == "XYZ" && num_dimensions != 3) {
    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
    return;
  }
  if (geometry_type == "XY" && num_dimensions != 2) {
    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    Log::error("XDMF geometry format not supported: " + format);
    return;
  }

  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset =
      h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
#ifndef NDEBUG
  H5T_class_t const type_class = dataset.getTypeClass();
  ASSERT(type_class == H5T_FLOAT);
#endif
  H5::FloatType const datatype = dataset.getFloatType();
  size_t const datatype_size = datatype.getSize();
#ifndef NDEBUG
  ASSERT(datatype_size == std::stoul(precision));
  H5::DataSpace const dataspace = dataset.getSpace();
  int const rank = dataspace.getSimpleExtentNdims();
  ASSERT(rank == 2);
  hsize_t dims[2];
  int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
  ASSERT(ndims == 2);
  ASSERT(dims[0] == static_cast<hsize_t>(num_verts));
  ASSERT(dims[1] == static_cast<hsize_t>(num_dimensions));
#endif
  if (datatype_size == 4) {
    addNodesToMesh<T, I, float>(soup, num_verts, num_dimensions, dataset, datatype,
                                geometry_type == "XYZ");
  } else if (datatype_size == 8) {
    addNodesToMesh<T, I, double>(soup, num_verts, num_dimensions, dataset, datatype,
                                 geometry_type == "XYZ");
  }
}

template <std::floating_point T, std::signed_integral I, std::signed_integral V>
static void
addElementsToMesh(Size const num_elements, String const & topology_type,
                  std::string const & dimensions, PolytopeSoup<T, I> & soup,
                  H5::DataSet const & dataset, H5::IntType const & datatype)
{
  if (topology_type == "Mixed") {
    // Expect dims to be one number
    auto const conn_length = sto<Size>(dimensions);
    Vector<V> data_vec(conn_length);
    dataset.read(data_vec.data(), datatype);
    // Add the elements to the soup
    Size position = 0;
    Size num_vertices = 0;
    Vector<I> conn;
    for (Size i = 0; i < num_elements; ++i) {
      auto const element_type = static_cast<int8_t>(data_vec[position]);
      VTKElemType const elem_type = xdmfToVTKElemType(element_type);
      auto const npoints = verticesPerElem(elem_type);
      if (npoints != num_vertices) {
        conn.resize(npoints);
        num_vertices = npoints;
      }
      for (Size j = 0; j < npoints; ++j) {
        conn[j] =
            static_cast<I>(static_cast<unsigned int>(data_vec[position + j + 1]));
      }
      position += npoints + 1;
      soup.addElement(elem_type, conn);
    }
  } else {
    size_t const split = dimensions.find_last_of(' ');
    auto const ncells = sto<Size>(dimensions.substr(0, split));
    auto const nverts = sto<Size>(dimensions.substr(split + 1));
    if (ncells != num_elements) {
      Log::error("Mismatch in number of elements");
      return;
    }
    Vector<V> data_vec(ncells * nverts);
    dataset.read(data_vec.data(), datatype);
    VTKElemType elem_type = VTKElemType::None;
    if (topology_type == "Triangle") {
      elem_type = VTKElemType::Triangle;
    } else if (topology_type == "Quadrilateral") {
      elem_type = VTKElemType::Quad;
    } else if (topology_type == "Triangle_6") {
      elem_type = VTKElemType::QuadraticTriangle;
    } else if (topology_type == "Quadrilateral_8") {
      elem_type = VTKElemType::QuadraticQuad;
    } else {
      Log::error("Unsupported element type");
    }
    Vector<I> conn(nverts);
    // Add the elements to the soup
    for (Size i = 0; i < ncells; ++i) {
      for (Size j = 0; j < nverts; ++j) {
        // NOLINTNEXTLINE
        conn[j] = static_cast<I>(data_vec[i * nverts + j]);
      }
      soup.addElement(elem_type, conn);
    }
  }
}

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void readXDMFTopology(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                             String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF topology");
  pugi::xml_node const xtopology = xgrid.child("Topology");
  if (strcmp(xtopology.name(), "Topology") != 0) {
    Log::error("XDMF topology node not found");
    return;
  }
  // Get the topology type
  String const topology_type(xtopology.attribute("TopologyType").value());
  // Get the number of elements
  Size const num_elements = sto<Size>(xtopology.attribute("NumberOfElements").value());
  // Get the DataItem node
  pugi::xml_node const xdataitem = xtopology.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    Log::error("XDMF topology DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Int") {
    Log::error("XDMF topology data type not supported: " + data_type);
    return;
  }
  // Get the precision
  std::string const precision(xdataitem.attribute("Precision").value());
  if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
    Log::error("XDMF topology precision not supported: " + String(precision.c_str()));
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    Log::error("XDMF geometry format not supported: " + format);
    return;
  }
  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset =
      h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
#ifndef NDEBUG
  H5T_class_t const type_class = dataset.getTypeClass();
  ASSERT(type_class == H5T_INTEGER);
#endif
  H5::IntType const datatype = dataset.getIntType();
  size_t const datatype_size = datatype.getSize();
#ifndef NDEBUG
  ASSERT(datatype_size == std::stoul(precision));
  H5::DataSpace const dataspace = dataset.getSpace();
  int const rank = dataspace.getSimpleExtentNdims();
  if (topology_type == "Mixed") {
    ASSERT(rank == 1);
    hsize_t dims[1];
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    ASSERT(ndims == 1);
  } else {
    ASSERT(rank == 2);
    hsize_t dims[2];
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    ASSERT(ndims == 2);
  }
#endif
  // Get the dimensions
  std::string const dimensions = xdataitem.attribute("Dimensions").value();
  if (datatype_size == 1) {
    addElementsToMesh<T, I, int8_t>(num_elements, topology_type, dimensions, soup,
                                    dataset, datatype);
  } else if (datatype_size == 2) {
    addElementsToMesh<T, I, int16_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else if (datatype_size == 4) {
    addElementsToMesh<T, I, int32_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else if (datatype_size == 8) {
    addElementsToMesh<T, I, int64_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else {
    Log::error("Unsupported data type size");
  }
}

//==============================================================================
// addElsetToMesh
//==============================================================================

template <std::floating_point T, std::signed_integral I, std::signed_integral V>
static void
addElsetToMesh(PolytopeSoup<T, I> & soup, Size const num_elements,
               H5::DataSet const & dataset, H5::IntType const & datatype,
               String const & elset_name)
{
  Vector<V> data_vec(num_elements);
  dataset.read(data_vec.data(), datatype);
  Vector<I> elset_ids(num_elements);
  for (Size i = 0; i < num_elements; ++i) {
    elset_ids[i] = static_cast<I>(static_cast<uint32_t>(data_vec[i]));
  }
  soup.addElset(elset_name, elset_ids);
}

//==============================================================================
// readXDMFElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void readXDMFElsets(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                           String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF elsets");
  // Loop over all nodes to find the elsets
  for (pugi::xml_node xelset = xgrid.first_child(); xelset;
       xelset = xelset.next_sibling()) {
    if (strcmp(xelset.name(), "Set") != 0) {
      continue;
    }
    // Get the SetType
    String const set_type(xelset.attribute("SetType").value());
    if (set_type != "Cell") {
      Log::error("XDMF elset only supports SetType=Cell");
      return;
    }
    // Get the name
    String const name(xelset.attribute("Name").value());
    if (name.size() == 0) {
      Log::error("XDMF elset name not found");
      return;
    }
    // Get the DataItem node
    pugi::xml_node const xdataitem = xelset.child("DataItem");
    if (strcmp(xdataitem.name(), "DataItem") != 0) {
      Log::error("XDMF elset DataItem node not found");
      return;
    }
    // Get the data type
    String const data_type(xdataitem.attribute("DataType").value());
    if (data_type != "Int") {
      Log::error("XDMF elset data type not supported: " + data_type);
      return;
    }
    // Get the precision
    std::string const precision = xdataitem.attribute("Precision").value();
    if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
      Log::error("XDMF elset precision not supported: " + String(precision.c_str()));
      return;
    }
    // Get the format
    String const format(xdataitem.attribute("Format").value());
    if (format != "HDF") {
      Log::error("XDMF elset format not supported: " + format);
      return;
    }
    // Get the h5 dataset path
    String const h5dataset(xdataitem.child_value());
    // Read the data
    H5::DataSet const dataset =
        h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
#ifndef NDEBUG
    H5T_class_t const type_class = dataset.getTypeClass();
    ASSERT(type_class == H5T_INTEGER);
#endif
    H5::IntType const datatype = dataset.getIntType();
    size_t const datatype_size = datatype.getSize();
    ASSERT(datatype_size == std::stoul(precision));
    H5::DataSpace const dataspace = dataset.getSpace();
#ifndef NDEBUG
    int const rank = dataspace.getSimpleExtentNdims();
    ASSERT(rank == 1);
#endif

    hsize_t dims[1];
#ifndef NDEBUG
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    ASSERT(ndims == 1);
    std::string const dimensions = xdataitem.attribute("Dimensions").value();
#else
    dataspace.getSimpleExtentDims(dims, nullptr);
#endif
    auto const num_elements = static_cast<Size>(dims[0]);
    ASSERT(num_elements == sto<Size>(dimensions));

    // Get the dimensions
    if (datatype_size == 1) {
      addElsetToMesh<T, I, int8_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 2) {
      addElsetToMesh<T, I, int16_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 4) {
      addElsetToMesh<T, I, int32_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 8) {
      addElsetToMesh<T, I, int64_t>(soup, num_elements, dataset, datatype, name);
    }
  }
}

//==============================================================================
// readXDMFUniformGrid
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
void readXDMFUniformGrid(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                         String const & h5filename, PolytopeSoup<T, I> & mesh)
{
  readXDMFGeometry(xgrid, h5file, h5filename, mesh);
  readXDMFTopology(xgrid, h5file, h5filename, mesh);
  readXDMFElsets(xgrid, h5file, h5filename, mesh);
}

//==============================================================================
// readXDMFFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
readXDMFFile(String const & filename, PolytopeSoup<T, I> & soup)
{
  Log::info("Reading XDMF mesh file: " + filename);

  // Open HDF5 file
  Size const h5filepath_end = filename.find_last_of('/') + 1;
  String const h5filename =
      filename.substr(h5filepath_end, filename.size() - 4 - h5filepath_end) + "h5";
  String const h5filepath = filename.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File h5file((h5filepath + h5filename).c_str(), H5F_ACC_RDONLY);

  // Setup XML file
  pugi::xml_document xdoc;
  pugi::xml_parse_result const result = xdoc.load_file(filename.c_str());
  if (!result) {
    Log::error("XDMF XML parse error: " + String(result.description()) +
               ", character pos= " + toString(result.offset));
  }
  pugi::xml_node const xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    Log::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node const xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    Log::error("XDMF XML domain node is not Domain");
    return;
  }

  pugi::xml_node const xgrid = xdomain.child("Grid");
  if (strcmp("Grid", xgrid.name()) != 0) {
    Log::error("XDMF XML grid node is not Grid");
    return;
  }
  if (strcmp("Uniform", xgrid.attribute("GridType").value()) == 0) {
    readXDMFUniformGrid(xgrid, h5file, h5filename, soup);
  } else if (strcmp("Tree", xgrid.attribute("GridType").value()) == 0) {
    Log::error("XDMF XML Tree is not supported");
  } else {
    Log::error("XDMF XML grid type is not Uniform or Tree");
  }
  // Close HDF5 file
  h5file.close();
  // Close XML file
  xdoc.reset();
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif

//==============================================================================
// IO
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
PolytopeSoup<T, I>::read(String const & filename)
{
  if (filename.ends_with(".inp")) {
    readAbaqusFile(filename, *this);
  } else if (filename.ends_with(".xdmf")) {
    readXDMFFile<T, I>(filename, *this);
  } else {
    Log::error("Unsupported file format.");
  }
}

template <std::floating_point T, std::signed_integral I>
void
PolytopeSoup<T, I>::write(String const & filename) const
{
  if (filename.ends_with(".xdmf")) {
    writeXDMF(filename);
  } else {
    Log::error("Unsupported file format.");
  }
}

} // namespace um2
