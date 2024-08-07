#include <um2/common/logger.hpp>
#include <um2/common/permutation.hpp>
#include <um2/common/strto.hpp>
#include <um2/config.hpp>
#include <um2/geometry/point.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/memory/addressof.hpp>
#include <um2/stdlib/numeric/iota.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/string_view.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>

#if UM2_USE_PUGIXML
#  include <pugixml.hpp>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>

// We dont have access to the header defining many of the HDF5 types, so we disable
// the clang-tidy warning.
// NOLINTBEGIN(misc-include-cleaner)
#include <concepts>

namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

PolytopeSoup::PolytopeSoup(String const & filename) { read(filename); }

//==============================================================================
// Getters
//==============================================================================

void
PolytopeSoup::getElement(Int const i, VTKElemType & type, Vector<Int> & conn) const
{
  ASSERT(i < _element_types.size());
  type = _element_types[i];
  auto const istart = _element_offsets[i];
  auto const iend = _element_offsets[i + 1];
  auto const n = iend - istart;
  conn.resize(n);
  for (Int j = 0; j < n; ++j) {
    conn[j] = _element_conn[istart + j];
  }
}

void
PolytopeSoup::getElset(Int const i, String & name, Vector<Int> & ids,
                       Vector<Float> & data) const
{
  ASSERT(i < _elset_names.size());
  name = _elset_names[i];
  auto const istart = _elset_offsets[i];
  auto const iend = _elset_offsets[i + 1];
  auto const n = iend - istart;
  ids.resize(n);
  um2::copy(_elset_ids.cbegin() + istart, _elset_ids.cbegin() + iend, ids.begin());
  if (!_elset_data[i].empty()) {
    data = _elset_data[i];
  }
}

void
PolytopeSoup::getElset(String const & name, Vector<Int> & ids, Vector<Float> & data) const
{
  for (auto const & elset_name : _elset_names) {
    if (elset_name == name) {
      auto const i = static_cast<Int>(&elset_name - _elset_names.data());
      auto const istart = _elset_offsets[i];
      auto const iend = _elset_offsets[i + 1];
      auto const n = iend - istart;
      ids.resize(n);
      um2::copy(_elset_ids.cbegin() + istart, _elset_ids.cbegin() + iend, ids.begin());
      if (!_elset_data[i].empty()) {
        data = _elset_data[i];
      }
      return;
    }
  }
  LOG_WARN("Elset ", name, " not found.");
}

//==============================================================================
// Modifiers
//==============================================================================

auto
PolytopeSoup::addVertex(Float x, Float y, Float z) -> Int
{
  _vertices.emplace_back(x, y, z);
  return _vertices.size() - 1;
}

auto
PolytopeSoup::addVertex(Point3F const & p) -> Int
{
  _vertices.emplace_back(p);
  return _vertices.size() - 1;
}

auto
PolytopeSoup::addElement(VTKElemType const type, Vector<Int> const & conn) -> Int
{
  _element_types.emplace_back(type);
  if (_element_offsets.empty()) {
    _element_offsets.emplace_back(0);
  }
  _element_offsets.emplace_back(_element_offsets.back() + conn.size());
  for (auto const & id : conn) {
    ASSERT(id < _vertices.size());
    _element_conn.emplace_back(id);
  }
  return _element_types.size() - 1;
}

auto
PolytopeSoup::addElset(String const & name, Vector<Int> const & ids,
                       Vector<Float> data) -> Int
{
  LOG_DEBUG("Adding elset: ", name);

  for (auto const & this_name : _elset_names) {
    if (this_name == name) {
      LOG_ERROR("Elset ", name, " already exists.");
      return -1;
    }
  }

  Int const num_ids = ids.size();
  if (num_ids == 0) {
    LOG_ERROR("Elset ids", name, " is empty.");
    return -1;
  }
  ASSERT(um2::is_sorted(ids.cbegin(), ids.cend()));

  if (!data.empty() && (data.size() != num_ids)) {
    LOG_ERROR("Elset data size does not match the number of ids.");
    return -1;
  }

  _elset_names.emplace_back(name);
  if (_elset_offsets.empty()) {
    _elset_offsets.emplace_back(0);
  }

  Int const old_num_ids = _elset_ids.size();
  Int const new_num_ids = old_num_ids + num_ids;
  _elset_offsets.emplace_back(new_num_ids);
  _elset_ids.resize(new_num_ids);
  um2::copy(ids.begin(), ids.end(), _elset_ids.data() + old_num_ids);
#if UM2_ENABLE_ASSERTS
  for (auto const & id : _elset_ids) {
    ASSERT(id < _element_types.size());
  }
#endif
  _elset_data.emplace_back(um2::move(data));
  return _elset_names.size() - 1;
}

void
PolytopeSoup::reserveMoreElements(VTKElemType const elem_type, Int const num_elems)
{
  // Element types
  _element_types.reserve(num_elems + _element_types.size());

  // Element offsets
  if (_element_offsets.empty()) {
    _element_offsets.reserve(num_elems + 1);
  } else {
    _element_offsets.reserve(num_elems + _element_offsets.size());
  }

  // Element connectivity
  Int const verts_per_elem = verticesPerElem(elem_type);
  _element_conn.reserve(num_elems * verts_per_elem + _element_conn.size());
}

void
PolytopeSoup::reserveMoreVertices(Int const num_verts)
{
  _vertices.reserve(num_verts + _vertices.size());
}

void
PolytopeSoup::sortElsets()
{
  // Create a vector that stores the name and index of each elset.
  using NameIndexPair = um2::Pair<String, Int>;
  Int const num_elsets = _elset_names.size();
  Vector<NameIndexPair> elset_name_index_pairs(num_elsets);
  for (Int i = 0; i < num_elsets; ++i) {
    elset_name_index_pairs[i] = NameIndexPair(_elset_names[i], i);
  }
  // Sort the vector by the elset names.
  std::sort(elset_name_index_pairs.begin(), elset_name_index_pairs.end());
  // Create new offsets, ids, and data vectors to hold the sorted elsets.
  Vector<Int> elset_offsets(_elset_offsets.size());
  Vector<Int> elset_ids(_elset_ids.size());
  Vector<Vector<Float>> elset_data(_elset_data.size());
  // For each elset, copy the data to its new location.
  Int offset = 0;
  elset_offsets[0] = offset;
  Vector<Int> perm;
  for (Int i = 0; i < num_elsets; ++i) {
    auto const & name = elset_name_index_pairs[i].first;
    auto const & index = elset_name_index_pairs[i].second;
    auto const iold = index;
    // The elset_name_index_pairs vector holds a copy of the elset names, so we
    // can overwrite the original elset name at the new index.
    _elset_names[i] = name;
    Int const len = _elset_offsets[iold + 1] - _elset_offsets[iold];
    ASSERT(len > 0);
    elset_offsets[i + 1] = offset + len;
    // copy the old elset ids to the new elset ids
    um2::copy(_elset_ids.begin() + _elset_offsets[iold],
              _elset_ids.begin() + _elset_offsets[iold + 1],
              elset_ids.begin() + elset_offsets[i]);
    // Sort the elset IDs and data
    perm.resize(len);
    sortPermutation(elset_ids.begin() + elset_offsets[i],
                    elset_ids.begin() + elset_offsets[i + 1], perm.begin());
    applyPermutation(elset_ids.begin() + elset_offsets[i],
                     elset_ids.begin() + elset_offsets[i + 1], perm.cbegin());
    // Move the old elset data to the new elset data
    elset_data[i] = um2::move(_elset_data[iold]);
    if (!elset_data[i].empty()) {
      applyPermutation(elset_data[i].begin(), elset_data[i].end(), perm.cbegin());
    }
    offset += len;
  }
  // Move the temporary vectors to the original vectors.
  _elset_offsets = um2::move(elset_offsets);
  _elset_ids = um2::move(elset_ids);
  _elset_data = um2::move(elset_data);
}

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PolytopeSoup::operator+=(PolytopeSoup const & other) noexcept -> PolytopeSoup &
{
  // Vertices
  auto const this_num_verts = numVertices();
  auto const other_num_verts = other.numVertices();
  _vertices.resize(this_num_verts + other_num_verts);
  um2::copy(other._vertices.cbegin(), other._vertices.cend(),
            _vertices.begin() + this_num_verts);

  // Elements types
  auto const this_num_elems = numElements();
  auto const other_num_elems = other.numElements();
  _element_types.resize(this_num_elems + other_num_elems);
  um2::copy(other._element_types.cbegin(), other._element_types.cend(),
            _element_types.begin() + this_num_elems);

  // Element offsets
  Int const last_offset = _element_offsets.empty() ? 0 : _element_offsets.back();
  _element_offsets.resize(this_num_elems + other_num_elems + 1);
  for (Int i = 0; i < other_num_elems + 1; ++i) {
    _element_offsets[this_num_elems + i] = other._element_offsets[i] + last_offset;
  }

  // Element connectivity
  auto const this_conn_size = _element_conn.size();
  auto const other_conn_size = other._element_conn.size();
  _element_conn.resize(this_conn_size + other_conn_size);
  for (Int i = 0; i < other_conn_size; ++i) {
    _element_conn[this_conn_size + i] = other._element_conn[i] + this_num_verts;
  }

  // Elsets
  if (other._elset_names.empty()) {
    return *this;
  }

  // Elset names
  auto const this_num_elsets = _elset_names.size();
  auto const other_num_elsets = other._elset_names.size();
  Vector<String> new_elset_names = _elset_names;
  new_elset_names.reserve(this_num_elsets + other_num_elsets);
  ASSERT(um2::is_sorted(_elset_names.cbegin(), _elset_names.cend()));
  ASSERT(um2::is_sorted(other._elset_names.cbegin(), other._elset_names.cend()));
  for (auto const & name : other._elset_names) {
    bool name_exists = false;
    for (auto const & this_name : _elset_names) {
      if (name == this_name) {
        name_exists = true;
        break;
      }
    }
    if (!name_exists) {
      new_elset_names.emplace_back(name);
    }
  }
  std::sort(new_elset_names.begin(), new_elset_names.end());

  // Elset offsets, ids, and data
  Vector<Int> new_elset_offsets(new_elset_names.size() + 1);
  new_elset_offsets[0] = 0;
  Vector<Int> new_elset_ids;
  new_elset_ids.reserve(_elset_ids.size() + other._elset_ids.size());
  Vector<Vector<Float>> new_elset_data;
  // For each elset
  for (Int i = 0; i < new_elset_names.size(); ++i) {
    auto const & name = new_elset_names[i];
    // Get the number of elements, ids, and data
    Int num_elements = 0;
    Vector<Float> data;
    // Loop over this elsets
    for (Int j = 0; j < this_num_elsets; ++j) {
      if (_elset_names[j] == name) {
        num_elements += _elset_offsets[j + 1] - _elset_offsets[j];
        for (Int k = _elset_offsets[j]; k < _elset_offsets[j + 1]; ++k) {
          new_elset_ids.emplace_back(_elset_ids[k]);
        }
        if (!_elset_data[j].empty()) {
          data = _elset_data[j];
        }
      }
    }
    // Loop over other elsets
    for (Int j = 0; j < other_num_elsets; ++j) {
      if (other._elset_names[j] == name) {
        num_elements += other._elset_offsets[j + 1] - other._elset_offsets[j];
        for (Int k = other._elset_offsets[j]; k < other._elset_offsets[j + 1]; ++k) {
          new_elset_ids.emplace_back(other._elset_ids[k] + this_num_elems);
        }
        auto const & other_data = other._elset_data[j];
        if (!other_data.empty()) {
          Int const old_size = data.size();
          data.resize(data.size() + other_data.size());
          um2::copy(other_data.cbegin(), other_data.cend(), data.begin() + old_size);
        }
      }
    }
    new_elset_offsets[i + 1] = new_elset_offsets[i] + num_elements;
    // Ids should already be sorted
    ASSERT(um2::is_sorted(new_elset_ids.cbegin() + new_elset_offsets[i],
                          new_elset_ids.cend()));
    new_elset_data.emplace_back(data);
  }

  _elset_names = um2::move(new_elset_names);
  _elset_offsets = um2::move(new_elset_offsets);
  _elset_ids = um2::move(new_elset_ids);
  _elset_data = um2::move(new_elset_data);
  return *this;
}

//==============================================================================
// compare
//==============================================================================

auto
PolytopeSoup::compare(PolytopeSoup const & other) const -> int
{

  // Compare vertices
  if (_vertices.size() != other._vertices.size()) {
    return 1;
  }
  for (Int i = 0; i < _vertices.size(); ++i) {
    if (!_vertices[i].isApprox(other._vertices[i])) {
      return 2;
    }
  }

  // Compare element types
  if (_element_types != other._element_types) {
    return 3;
  }

  // Compare element offsets
  if (_element_offsets != other._element_offsets) {
    return 4;
  }

  // Compare element connectivity
  if (_element_conn != other._element_conn) {
    return 5;
  }

  // Compare elset names
  if (_elset_names != other._elset_names) {
    return 6;
  }

  // Compare elset offsets
  if (_elset_offsets != other._elset_offsets) {
    return 7;
  }

  // Compare elset ids
  if (_elset_ids != other._elset_ids) {
    return 8;
  }

  // Compare elset data
  if (_elset_data.size() != other._elset_data.size()) {
    return 9;
  }
  // Avoid exact comparison of floating point data
  if (!_elset_data.empty()) {
    for (Int i = 0; i < _elset_data.size(); ++i) {
      if (_elset_data[i].size() != other._elset_data[i].size()) {
        return 10;
      }
      for (Int j = 0; j < _elset_data[i].size(); ++j) {
        if (um2::abs(_elset_data[i][j] - other._elset_data[i][j]) >
            epsDistance<Float>()) {
          return 11;
        }
      }
    }
  }
  return 0;
}

//==============================================================================
// getSubset
//==============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PolytopeSoup::getSubset(String const & elset_name, PolytopeSoup & subset) const
{
  LOG_DEBUG("Extracting subset: ", elset_name);

  // Find the elset with the given name.
  Int elset_index = 0;
  bool found = false;
  for (Int i = 0; i < _elset_names.size(); ++i) {
    if (_elset_names[i] == elset_name) {
      elset_index = i;
      found = true;
      break;
    }
  }
  if (!found) {
    LOG_ERROR("getSubset: Elset '", elset_name, "' not found");
    return;
  }

  // Get the element ids of the subset.
  auto const subset_elset_start = _elset_offsets[elset_index];
  auto const subset_elset_end = _elset_offsets[elset_index + 1];
  auto const subset_num_elements = subset_elset_end - subset_elset_start;
  Vector<Int> const element_ids(_elset_ids.cbegin() + subset_elset_start,
                                _elset_ids.cbegin() + subset_elset_end);

  if (!um2::is_sorted(element_ids.cbegin(), element_ids.cend())) {
    LOG_ERROR("getSubset: Elset IDs are not sorted. Use sortElsets() to correct this.");
    return;
  }

  // Get the length of the subset element connectivity as well as the number of each
  // element type.
  Vector<um2::Pair<VTKElemType, Int>> subset_elem_type_counts;
  Int subset_element_conn_len = 0;
  for (Int i = 0; i < subset_num_elements; ++i) {
    auto const element_id = element_ids[i];
    auto const element_start = _element_offsets[element_id];
    auto const element_end = _element_offsets[element_id + 1];
    auto const element_len = element_end - element_start;
    subset_element_conn_len += element_len;
    auto const element_type = _element_types[element_id];
    found = false;
    for (auto & type_count : subset_elem_type_counts) {
      auto const type = type_count.first;
      if (type == element_type) {
        ++type_count.second;
        found = true;
        break;
      }
    }
    if (!found) {
      subset_elem_type_counts.emplace_back(element_type, 1);
    }
  }

  // Get the vertex ids of the subset.
  Vector<Int> vertex_ids;
  vertex_ids.reserve(subset_element_conn_len);
  for (Int i = 0; i < subset_num_elements; ++i) {
    auto const element_id = element_ids[i];
    auto const element_start = _element_offsets[element_id];
    auto const element_end = _element_offsets[element_id + 1];
    auto const element_len = element_end - element_start;
    for (Int j = 0; j < element_len; ++j) {
      Int const vertex_id = _element_conn[element_start + j];
      vertex_ids.emplace_back(vertex_id);
    }
  }

  // Get the unique vertex ids of the subset.
  std::sort(vertex_ids.begin(), vertex_ids.end());
  Vector<Int> unique_vertex_ids = vertex_ids;
  auto const * const last =
      std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
  auto const num_unique_verts = static_cast<Int>(last - unique_vertex_ids.cbegin());

  // Add each of the unique vertices to the subset.
  subset.reserveMoreVertices(num_unique_verts);
  for (Int i = 0; i < num_unique_verts; ++i) {
    auto const vertex_id = unique_vertex_ids[i];
    auto const & vertex = _vertices[vertex_id];
    subset.addVertex(vertex);
  }

  // Reserve space for elements
  for (auto const & type_count : subset_elem_type_counts) {
    VTKElemType const type = type_count.first;
    Int const count = type_count.second;
    subset.reserveMoreElements(type, count);
  }

  // For each element, add it to the subset, remapping the vertex IDs
  // unique_vertex_ids[new_id] = old_id
  Vector<Int> conn;
  for (Int i = 0; i < subset_num_elements; ++i) {
    auto const element_id = element_ids[i];
    auto const element_type = _element_types[element_id];
    auto const element_start = _element_offsets[element_id];
    auto const element_end = _element_offsets[element_id + 1];
    auto const element_len = element_end - element_start;
    conn.resize(element_len);
    for (Int j = 0; j < element_len; ++j) {
      Int const old_vertex_id = _element_conn[element_start + j];
      auto const * const it =
          std::lower_bound(unique_vertex_ids.cbegin(), last, old_vertex_id);
      auto const new_vertex_id = static_cast<Int>(it - unique_vertex_ids.cbegin());
      ASSERT(*it == old_vertex_id);
      conn[j] = new_vertex_id;
    }
    subset.addElement(element_type, conn);
  }

  // If the intersection of this elset and another elset is non-empty, then we need to
  // add the itersection as an elset and remap the elset IDs using the element_ids
  // vector.
  //
  // element_ids[i] is the old element id, and i is the new element id.
  // The largest possible intersection is the size of the elset itself.

  Vector<Int> ids;
  Vector<Float> elset_data;
  Int * intersection = new Int[static_cast<size_t>(subset_num_elements)];
  Int const num_elsets = _elset_names.size();
  for (Int i = 0; i < num_elsets; ++i) {
    if (i == elset_index) {
      continue;
    }
    auto const elset_start = _elset_offsets[i];
    auto const elset_end = _elset_offsets[i + 1];
    auto const * const elset_ids_begin = addressof(_elset_ids[elset_start]);
    auto const * const elset_ids_end = elset_ids_begin + (elset_end - elset_start);
    ASSERT(um2::is_sorted(elset_ids_begin, elset_ids_end));
    Int * intersection_end =
        std::set_intersection(element_ids.begin(), element_ids.end(), elset_ids_begin,
                              elset_ids_end, intersection);
    // No intersection
    if (intersection_end == intersection) {
      continue;
    }
    auto const & name = _elset_names[i];
    auto const num_ids = static_cast<Int>(intersection_end - intersection);
    ids.resize(num_ids);
    // Remap the element IDs
    for (Int j = 0; j < num_ids; ++j) {
      Int const old_element_id = intersection[j];
      auto const * const it =
          std::lower_bound(element_ids.cbegin(), element_ids.cend(), old_element_id);
      auto const new_element_id = static_cast<Int>(it - element_ids.cbegin());
      ASSERT(*it == old_element_id);
      ids[j] = new_element_id;
    }
    if (_elset_data[i].empty()) {
      subset.addElset(name, ids);
      continue;
    }
    // There is data
    elset_data.resize(num_ids);
    Vector<Float> const & this_elset_data = _elset_data[i];
    for (Int j = 0; j < num_ids; ++j) {
      Int const old_element_id = intersection[j];
      auto const * const it =
          std::lower_bound(elset_ids_begin, elset_ids_end, old_element_id);
      ASSERT(*it == old_element_id);
      auto const idx = static_cast<Int>(it - elset_ids_begin);
      elset_data[j] = this_elset_data[idx];
    }
    subset.addElset(name, ids, elset_data);
  }
  delete[] intersection;
}

//==============================================================================-
// IO for ABAQUS files.
//==============================================================================

namespace
{

auto
abaqusParseNodes(PolytopeSoup & soup, std::ifstream & file, char * const line,
                 uint64_t const max_line_length) -> StringView
{
  // line starts with "*NODE"
  auto const smax_line_length = static_cast<int64_t>(max_line_length);
  while (file.getline(line, smax_line_length) && line[0] != '*') {
    StringView line_view(line);
    // Format: node_id, x, y, z
    // Skip ID
    StringView token = line_view.getTokenAndShrink(',');
    ASSERT(!token.empty());

    char * end = nullptr;

    // x
    token = line_view.getTokenAndShrink(',');
    ASSERT(!token.empty());
    Float const x = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;

    // y
    token = line_view.getTokenAndShrink(',');
    ASSERT(!token.empty());
    Float const y = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;

    // Only final token left of the form " zzzz\n"
    Float const z = strto<Float>(line_view.data(), &end);
    ASSERT(end != nullptr);
    soup.addVertex(x, y, z);
  }
  ASSERT(file.peek() != EOF);
  return {line};
} // abaqusParseNodes

auto
abaqusParseElements(PolytopeSoup & soup, std::ifstream & file, char * const line,
                    uint64_t const max_line_length) -> StringView
{
  //  "*ELEMENT, type=CPS" is 18 characters
  //  CPS3 is a 3-node triangle
  //  CPS4 is a 4-node quadrilateral
  //  CPS6 is a 6-node quadratic triangle
  //  CPS8 is a 8-node quadratic quadrilateral
  //  Hence, line[18] is the offset of the element type
  //  ASCII code for '0' is 48, so line[18] - 48 is the offset
  //  as an integer
  //
#if UM2_ENABLE_ASSERTS
  StringView const info_line_view(line);
  ASSERT(info_line_view.starts_with("*ELEMENT, type=CPS"));
  ASSERT(info_line_view.size() > 18);
#endif
  if (line[15] != 'C' || line[16] != 'P' || line[17] != 'S') {
    LOG_ERROR("Only CPS elements are supported");
    return {};
  }
  Int const offset = static_cast<Int>(line[18]) - 48;
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
    LOG_ERROR("AbaqusCellType CPS", offset, " is not supported");
    break;
  }
  }
  Int const verts_per_elem = verticesPerElem(this_type);
  Vector<Int> conn(verticesPerElem(this_type));
  auto const smax_line_length = static_cast<int64_t>(max_line_length);
  while (file.getline(line, smax_line_length) && line[0] != '*') {
    StringView line_view(line);
    // For each element, read the element ID and the node IDs
    // Format: id, n1, n2, n3, n4, n5 ...
    // Skip ID
    StringView token = line_view.getTokenAndShrink(',');
    ASSERT(!token.empty());

    for (Int i = 0; i < verts_per_elem; ++i) {
      token = line_view.getTokenAndShrink(',');
      ASSERT(!token.empty());
      char * end = nullptr;
      Int const id = strto<Int>(token.data(), &end);
      ASSERT(end != nullptr);
      ASSERT(id > 0);
      conn[i] = id - 1; // ABAQUS is 1-indexed
    }
    soup.addElement(this_type, conn);
  }

  if (file.peek() == EOF) {
    return {};
  }
  return {line};
} // abaqusParseElements

auto
abaqusParseElsets(PolytopeSoup & soup, std::ifstream & file, char * const line,
                  uint64_t const max_line_length) -> StringView
{
  StringView const info_line_view(line);
  ASSERT(info_line_view.starts_with("*ELSET,ELSET="));
  ASSERT(info_line_view.size() > 13);
  String const elset_name(info_line_view.substr(13, info_line_view.size() - 13));
  Vector<Int> elset_ids;
  auto const smax_line_length = static_cast<int64_t>(max_line_length);
  while (file.getline(line, smax_line_length) && line[0] != '*') {
    StringView line_view(line);
    // Add each element ID to the elset
    // Format: id, id, id, id, id,
    // Note, line ends in ", " or ",", hence a meaningful token is always followed
    // by a comma and is at least 2 characters long: ex "1," or "1, "
    uint64_t comma = line_view.find_first_of(',');
    ASSERT(comma > 0);
    while (comma != StringView::npos) {
      // token = "id"
      StringView const token = line_view.substr(0, comma);
      char * end = nullptr;
      Int const id = strto<Int>(token.data(), &end);
      ASSERT(end != nullptr);
      ASSERT(id > 0);
      elset_ids.emplace_back(id - 1); // ABAQUS is 1-indexed

      // Done if there are not enough characters left for another id
      if (line_view.size() <= comma + 2) {
        break;
      }
      // Otherwise, remove the token and the comma, and find the next comma
      line_view.remove_prefix(comma + 1);
      comma = line_view.find_first_of(',');
    }
  }
  ASSERT(um2::is_sorted(elset_ids.cbegin(), elset_ids.cend()));
  soup.addElset(elset_name, elset_ids);
  if (file.peek() == EOF) {
    return {};
  }
  return {line};
} // abaqusParseElsets

void
readAbaqusFile(String const & filename, PolytopeSoup & soup)
{
  LOG_INFO("Reading Abaqus file: ", filename);

  uint64_t constexpr max_line_length = 1024;
  char line[max_line_length];

  // Open file
  std::ifstream file(filename.data());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: ", filename);
    return;
  }

  // General structure of an Abaqus file:
  // *Heading
  //  mesh_name
  // *NODE
  //  node_id, x, y, z
  //  ...
  // *ELEMENT, type=....
  // id, n1, n2, n3, n4, n5 ...
  // ...
  // *ELEMENT, type=....
  // id, n1, n2, n3, n4, n5 ...
  // ...
  // *ELSET,ELSET=....
  // id, id, id, id, id, ...
  // ...
  // *ELSET,ELSET=....
  // id, id, id, id, id, ...
  // ...
  //
  // Additionally, there may be comments which start with "**".

  // Get the first line
  file.getline(line, max_line_length);
  StringView line_view(line);

  bool get_next_line = false;
  while (file.peek() != EOF) {
    // Only get the next line if it does not start with *NODE, *ELEMENT, or *ELSET
    if (get_next_line) {
      file.getline(line, max_line_length);
      line_view = StringView(line);
    }
    get_next_line = true;
    while (line_view.starts_with("*NODE")) {
      line_view = abaqusParseNodes(soup, file, line, max_line_length);
      get_next_line = false;
    }
    while (line_view.starts_with("*ELEMENT")) {
      line_view = abaqusParseElements(soup, file, line, max_line_length);
      get_next_line = false;
    }
    while (line_view.starts_with("*ELSET")) {
      line_view = abaqusParseElsets(soup, file, line, max_line_length);
      get_next_line = false;
    }
  }

  if (soup.numElsets() > 0) {
    soup.sortElsets();
  }

  file.close();
  LOG_INFO("Finished reading Abaqus file: ", filename);
} // readAbaqusFile

} // namespace

//=============================================================================
// IO for VTK files
//=============================================================================

namespace
{

void
vtkParseUnstructuredGrid(PolytopeSoup & soup, std::ifstream & file, char * const line,
                         uint64_t const max_line_length)
{
  // line starts with "UNSTRUCTURED_GRID"
  auto const smax_line_length = static_cast<int64_t>(max_line_length);

  // Get the number of points
  file.getline(line, smax_line_length);
  StringView line_view(line);
  if (!line_view.starts_with("POINTS")) {
    LOG_ERROR("Expected POINTS");
    return;
  }
  line_view.getTokenAndShrink(); // Remove "POINTS"
  StringView token = line_view.getTokenAndShrink();
  ASSERT(!token.empty());
  char * end = nullptr;
  Int const num_points = strto<Int>(token.data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  ASSERT(num_points > 0);
  soup.reserveMoreVertices(num_points);

  // line_view now just contains the data type
  ASSERT(line_view.starts_with("float") || line_view.starts_with("double"));

  // Read the vertices
  while (file.getline(line, smax_line_length)) {
    if (line[0] == '\0') {
      continue;
    }
    line_view = StringView(line);
    if (line_view.starts_with("CELLS")) {
      break;
    }
    // Format: x y z
    // x
    token = line_view.getTokenAndShrink();
    ASSERT(!token.empty());
    Float const x = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;

    // y
    token = line_view.getTokenAndShrink();
    ASSERT(!token.empty());
    Float const y = strto<Float>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;

    // Only final token left of the form " zzzz\n"
    Float const z = strto<Float>(line_view.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    soup.addVertex(x, y, z);
  }

  // Line starts with "CELLS"
  ASSERT(line_view.starts_with("CELLS"));
  line_view.getTokenAndShrink(); // Remove "CELLS"
  // Get the number of cells and the length of the cell connectivity
  token = line_view.getTokenAndShrink();
  ASSERT(!token.empty());
  Int const num_cells = strto<Int>(token.data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  ASSERT(num_cells > 0);
#if UM2_ENABLE_ASSERTS
  token = line_view.getTokenAndShrink();
  ASSERT(!token.empty());
  Int const cell_conn_len = strto<Int>(token.data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  ASSERT(cell_conn_len > 0);
#else
  line_view.getTokenAndShrink();
#endif

  // Read the cells
  // Since we only accept a few cell types, we try to infer the cell type from the
  // number of vertices per cell.
  Vector<Int> conn;
  for (Int icell = 0; icell < num_cells; ++icell) {
    file.getline(line, smax_line_length);
    line_view = StringView(line);
    // Format: num_verts v1 v2 v3 ...
    // num_verts
    token = line_view.getTokenAndShrink();
    ASSERT(!token.empty());
    Int const num_verts = strto<Int>(token.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    ASSERT(num_verts > 0);
    conn.resize(num_verts);
    for (Int i = 0; i < num_verts; ++i) {
      token = line_view.getTokenAndShrink();
      ASSERT(!token.empty());
      Int const id = strto<Int>(token.data(), &end);
      ASSERT(end != nullptr);
      ASSERT(id >= 0);
      ASSERT(id < num_points);
      conn[i] = id;
    }
    VTKElemType const type = inferVTKElemType(num_verts);
    ASSERT(type != VTKElemType::Invalid);
    soup.addElement(type, conn);
  }

  // Skip the blank line
  file.getline(line, smax_line_length);
  ASSERT(line[0] == '\0');
  file.getline(line, smax_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("CELL_TYPES")) {
    LOG_ERROR("Expected CELL_TYPES");
    return;
  }

  // Line starts with "CELL_TYPES"
  // The next line is potentially VERY large, since MPACT outputs the cell type
  // array with a simple WRITE(*,*) type of statement. We skip it.
  file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  ASSERT(file.good());
} // vtkParseUnstructuredGrid

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
vtkParseCellData(PolytopeSoup & soup, std::ifstream & file, char * const line,
                 uint64_t const max_line_length)
{
  // line starts with "CELL_DATA"
  auto const smax_line_length = static_cast<int64_t>(max_line_length);

  StringView line_view(line);
  if (!line_view.starts_with("CELL_DATA")) {
    LOG_ERROR("Expected CELL_DATA");
    return;
  }

  // Get the number of cells in the dataset. It should be
  // equal to the number of elements in the soup.
  line_view.getTokenAndShrink(); // Remove "CELL_DATA"
  char * end = nullptr;
  Int const num_data = strto<Int>(line_view.data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  ASSERT(num_data == soup.numElements());

  // Ensure SCALARS data
  file.getline(line, smax_line_length);
  line_view = StringView(line);
  ASSERT(line_view.starts_with("SCALARS"));

  // Get the name of the data set
  StringView token = line_view.getTokenAndShrink();
  token = line_view.getTokenAndShrink();
  String const data_name(token);
  bool const is_float = line_view.starts_with("float") || line_view.starts_with("double");
  if (!is_float && !line_view.starts_with("int")) {
    LOG_ERROR("Only float, double, and int data types are supported");
    return;
  }

  // Ensure LOOKUP_TABLE
  file.getline(line, smax_line_length);
  ASSERT(StringView(line).starts_with("LOOKUP_TABLE"));

  // Read the data
  Vector<Float> data(num_data);
  Vector<Int> ids(num_data);
  um2::iota(ids.begin(), ids.end(), 0);
  Int data_ctr = 0;
  while (file.getline(line, smax_line_length) && line[0] != '\0') {
    line_view = StringView(line);
    token = line_view.getTokenAndShrink();
    while (!token.empty()) {
      Float const value = strto<Float>(token.data(), &end);
      ASSERT(end != nullptr);
      end = nullptr;
      // if the data type is not float, ensure the conversion was lossless
      if (!is_float) {
        int64_t const ivalue = strto<int64_t>(token.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
        if constexpr (std::same_as<Float, double>) {
          // If Float == double, then the significand is 52 bits (excluding the sign bit)
          // Hence, we can safely convert all integers less than 2^52 to double
          int64_t constexpr max_int = 1LL << 52;
          if (um2::abs(ivalue) > max_int) {
            LOG_ERROR("Integer value ", ivalue, " is too large to be stored as a double");
            return;
          }
        } else {
          // If Float == float, then the significand is 23 bits (excluding the sign bit)
          // Hence, we can safely convert all integers less than 2^23 to float
          int64_t constexpr max_int = 1LL << 23;
          if (um2::abs(ivalue) > max_int) {
            LOG_ERROR("Integer value ", ivalue, " is too large to be stored as a float");
            return;
          }
        }
      }
      data[data_ctr] = value;
      ++data_ctr;
      token = line_view.getTokenAndShrink();
    }
  }
  ASSERT(data_ctr == num_data);
  // Add elset
  soup.addElset(data_name, ids, data);
}

void
readVTKFile(String const & filename, PolytopeSoup & soup)
{
  LOG_INFO("Reading VTK file: ", filename);

  uint64_t constexpr max_line_length = 1024;
  char line[max_line_length];

  // Open file
  std::ifstream file(filename.data());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: ", filename);
    return;
  }

  // General structure of a VTK file:
  // # vtk DataFile Version 3.0
  // header
  // ASCII | BINARY (only support ASCII)
  // DATASET data_set_type (only support UNSTRUCTURED_GRID)
  // ...
  // POINT_DATA num_points
  // ...
  // CELL_DATA num_cells
  // ...

  // Get the first line and check the version
  file.getline(line, max_line_length);
  StringView line_view(line);
  if (!line_view.starts_with("# vtk DataFile Version 3.0")) {
    LOG_ERROR("Only VTK files of version 3.0 are supported");
    return;
  }

  // Skip the the header
  file.getline(line, max_line_length);

  // Get the file format
  file.getline(line, max_line_length);
  line_view = StringView(line);
  if (!line_view.starts_with("ASCII")) {
    LOG_ERROR("Only ASCII VTK files are supported");
    return;
  }

  // Get the dataset type
  file.getline(line, max_line_length);
  line_view = StringView(line);
  ASSERT(line_view.starts_with("DATASET"));
  // getTokenAndShrink() will remove "DATASET " from line_view
  line_view.getTokenAndShrink();
  if (line_view.starts_with("UNSTRUCTURED_GRID")) {
    vtkParseUnstructuredGrid(soup, file, line, max_line_length);
  } else {
    LOG_ERROR("Unsupported VTK dataset type: ", line_view);
    return;
  }

  // If there are any lines left, it's data. We only support scalar CELL_DATA of
  // float or double type
  if (file.peek() != EOF) {
    while (file.getline(line, max_line_length)) {
      if (line[0] == '\0') {
        continue;
      }
      line_view = StringView(line);
      if (line_view.starts_with("CELL_DATA")) {
        vtkParseCellData(soup, file, line, max_line_length);
      }
    }
  }

  if (soup.numElsets() > 0) {
    soup.sortElsets();
  }

  file.close();
  LOG_INFO("Finished reading VTK file: ", filename);
} // readVTKFile

} // namespace

#if UM2_HAS_XDMF
//==============================================================================
// IO for XDMF files
//==============================================================================

namespace
{

template <typename T>
inline auto
getH5DataType() -> H5::PredType
{
  if constexpr (std::same_as<T, float>) {
    return H5::PredType::NATIVE_FLOAT;
  }
  if constexpr (std::same_as<T, double>) {
    return H5::PredType::NATIVE_DOUBLE;
  }
  if constexpr (std::same_as<T, int8_t>) {
    return H5::PredType::NATIVE_INT8;
  }
  if constexpr (std::same_as<T, int16_t>) {
    return H5::PredType::NATIVE_INT16;
  }
  if constexpr (std::same_as<T, int32_t>) {
    return H5::PredType::NATIVE_INT32;
  }
  if constexpr (std::same_as<T, int64_t>) {
    return H5::PredType::NATIVE_INT64;
  }
  if constexpr (std::same_as<T, uint8_t>) {
    return H5::PredType::NATIVE_UINT8;
  }
  if constexpr (std::same_as<T, uint16_t>) {
    return H5::PredType::NATIVE_UINT16;
  }
  if constexpr (std::same_as<T, uint32_t>) {
    return H5::PredType::NATIVE_UINT32;
  }
  if constexpr (std::same_as<T, uint64_t>) {
    return H5::PredType::NATIVE_UINT64;
  }
  ASSERT(false);
  return H5::PredType::NATIVE_FLOAT;
}

void
writeXDMFGeometry(pugi::xml_node & xgrid, H5::Group & h5group, String const & h5filename,
                  String const & h5path, PolytopeSoup const & soup,
                  Point3F const & origin)
{
  Int const num_verts = soup.numVertices();
  auto const & vertices = soup.vertices();
  bool const soup_is_3d =
      std::any_of(vertices.cbegin(), vertices.cend(),
                  [](auto const & v) { return um2::abs(v[2]) > epsDistance<Float>(); });
  bool const origin_is_3d = um2::abs(origin[2]) > epsDistance<Float>();
  Int const dim = (soup_is_3d || origin_is_3d) ? 3 : 2;
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
  xdata.append_attribute("Dimensions") = (String(num_verts) + " " + String(dim)).data();
  xdata.append_attribute("Precision") = sizeof(Float);
  xdata.append_attribute("Format") = "HDF";
  String const h5geompath = h5filename + ":" + h5path + "/Geometry";
  xdata.append_child(pugi::node_pcdata).set_value(h5geompath.data());

  // Create HDF5 data space
  hsize_t dims[2] = {static_cast<hsize_t>(num_verts), static_cast<hsize_t>(dim)};
  H5::DataSpace const h5space(2, dims);
  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<Float>();
  // Create HDF5 data set
  H5::DataSet const h5dataset = h5group.createDataSet("Geometry", h5type, h5space);
  // Create an xy or xyz array
  Vector<Float> xyz(num_verts * dim);
  if (dim == 2) {
    for (Int i = 0; i < num_verts; ++i) {
      xyz[2 * i] = vertices[i][0] + origin[0];
      xyz[2 * i + 1] = vertices[i][1] + origin[1];
    }
  } else { // dim == 3
    for (Int i = 0; i < num_verts; ++i) {
      xyz[3 * i] = vertices[i][0] + origin[0];
      xyz[3 * i + 1] = vertices[i][1] + origin[1];
      xyz[3 * i + 2] = vertices[i][2] + origin[2];
    }
  }
  // Write HDF5 data set
  h5dataset.write(xyz.data(), h5type, h5space);
} // writeXDMFgeometry

void
writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group, String const & h5filename,
                  String const & h5path, PolytopeSoup const & soup)
{
  // Create XDMF Topology node
  auto xtopo = xgrid.append_child("Topology");
  Int const nelems = soup.numElements();

  Vector<Int> topology;
  String topology_type;
  String dimensions;
  Int nverts = 0;
  auto const elem_type = soup.getElemTypes();
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
      logger::error("Unsupported polytope type");
      return;
    }
    dimensions = String(nelems) + " " + String(nverts);
  } else {
    topology_type = "Mixed";
    ishomogeneous = false;
    auto const & element_conn = soup.elementConnectivity();
    auto const & element_offsets = soup.elementOffsets();
    auto const & element_types = soup.elementTypes();
    dimensions = String(nelems + element_conn.size());
    topology.resize(nelems + element_conn.size());
    // Create the topology array (type id + node ids)
    Int topo_ctr = 0;
    for (Int i = 0; i < nelems; ++i) {
      auto const topo_type = static_cast<int8_t>(vtkToXDMFElemType(element_types[i]));
      if (topo_type == static_cast<int8_t>(XDMFElemType::Invalid)) {
        logger::error("Unsupported polytope type");
        return;
      }
      ASSERT(topo_type > 0);
      topology[topo_ctr] = static_cast<Int>(static_cast<uint32_t>(topo_type));
      auto const offset = element_offsets[i];
      auto const npts = element_offsets[i + 1] - element_offsets[i];
      for (Int j = 0; j < npts; ++j) {
        topology[topo_ctr + j + 1] = element_conn[offset + j];
      }
      topo_ctr += npts + 1;
    }
  }
  xtopo.append_attribute("TopologyType") = topology_type.data();
  xtopo.append_attribute("NumberOfElements") = nelems;
  // Create XDMF DataItem node
  auto xdata = xtopo.append_child("DataItem");
  xdata.append_attribute("DataType") = "Int";
  xdata.append_attribute("Dimensions") = dimensions.data();
  xdata.append_attribute("Precision") = sizeof(Int);
  xdata.append_attribute("Format") = "HDF";
  String const h5topopath = h5filename + ":" + h5path + "/Topology";
  xdata.append_child(pugi::node_pcdata).set_value(h5topopath.data());

  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<Int>();
  if (ishomogeneous) {
    // Create HDF5 data space
    hsize_t dims[2] = {static_cast<hsize_t>(nelems), static_cast<hsize_t>(nverts)};
    H5::DataSpace const h5space(2, dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(soup.elementConnectivity().data(), h5type, h5space);
  } else {
    // Create HDF5 data space
    auto const dims = static_cast<hsize_t>(topology.size());
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(topology.data(), h5type, h5space);
  }
} // writeXDMFTopology

void
writeXDMFElsets(pugi::xml_node & xgrid, H5::Group & h5group, String const & h5filename,
                String const & h5path, PolytopeSoup const & soup)
{
  auto const & elset_names = soup.elsetNames();
  auto const & elset_offsets = soup.elsetOffsets();
  auto const & elset_ids = soup.elsetIDs();
  auto const & elset_data = soup.elsetData();
  for (Int i = 0; i < elset_names.size(); ++i) {
    String const & name = elset_names[i];
    auto const start = elset_offsets[i];
    auto const end = elset_offsets[i + 1];
    // Create HDF5 data space
    auto dims = static_cast<hsize_t>(end - start);
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data type
    H5::DataType const h5type = getH5DataType<Int>();
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet(name.data(), h5type, h5space);
    // Write HDF5 data set.
    h5dataset.write(&(elset_ids[start]), h5type, h5space);

    // Create XDMF Elset node
    auto xelset = xgrid.append_child("Set");
    xelset.append_attribute("Name") = name.data();
    xelset.append_attribute("SetType") = "Cell";
    // Create XDMF DataItem node
    auto xdata = xelset.append_child("DataItem");
    xdata.append_attribute("DataType") = "Int";
    xdata.append_attribute("Dimensions") = end - start;
    xdata.append_attribute("Precision") = sizeof(Int);
    xdata.append_attribute("Format") = "HDF";
    String const h5elsetpath = h5filename + ":" + h5path + "/" + name;
    xdata.append_child(pugi::node_pcdata).set_value(h5elsetpath.data());

    if (!elset_data[i].empty()) {
      // Create HDF5 data space
      auto const dims_data = static_cast<hsize_t>(elset_data[i].size());
      H5::DataSpace const h5space_data(1, &dims_data);
      // Create HDF5 data type
      H5::DataType const h5type_data = getH5DataType<Float>();
      // Create HDF5 data set
      H5::DataSet const h5dataset_data =
          h5group.createDataSet((name + "_data").data(), h5type_data, h5space_data);
      // Write HDF5 data set
      h5dataset_data.write(elset_data[i].data(), h5type_data, h5space_data);

      // Create XDMF data node
      auto xatt = xelset.append_child("Attribute");
      xatt.append_attribute("Name") = (name + "_data").data();
      xatt.append_attribute("Center") = "Cell";
      // Create XDMF DataItem node
      auto xdata2 = xatt.append_child("DataItem");
      xdata2.append_attribute("DataType") = "Float";
      xdata2.append_attribute("Dimensions") = elset_data[i].size();
      xdata2.append_attribute("Precision") = sizeof(Float);
      xdata2.append_attribute("Format") = "HDF";

      String const h5elsetdatapath = h5elsetpath + "_data";
      xdata2.append_child(pugi::node_pcdata).set_value(h5elsetdatapath.data());
    }
  }
} // writeXDMFelsets

} // namespace

void
writeXDMFUniformGrid(String const & name, pugi::xml_node & xdomain, H5::H5File & h5file,
                     String const & h5filename, String const & h5path,
                     PolytopeSoup const & soup, Point3F const & origin)
{
  // Grid
  pugi::xml_node xgrid = xdomain.append_child("Grid");
  xgrid.append_attribute("Name") = name.data();
  xgrid.append_attribute("GridType") = "Uniform";

  // h5
  String const h5grouppath = h5path + "/" + name;
  H5::Group h5group = h5file.createGroup(h5grouppath.data());

  writeXDMFGeometry(xgrid, h5group, h5filename, h5grouppath, soup, origin);
  writeXDMFTopology(xgrid, h5group, h5filename, h5grouppath, soup);
  writeXDMFElsets(xgrid, h5group, h5filename, h5grouppath, soup);
} // writeXDMFUniformGrid

namespace
{

void
writeXDMFFile(String const & filepath, PolytopeSoup const & soup)
{
  LOG_INFO("Writing XDMF file: ", filepath);
  ASSERT(filepath.ends_with(".xdmf"));

  // Setup HDF5 file
  // Get the h5 file name and path separately
  Int last_slash = filepath.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = 0;
  }
  // If there is no slash, the file name and path are the same
  // If there is a slash, the file name is everything after the last slash
  // and the path is everything before and including the last slash
  Int const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
  ASSERT(h5filepath_end < filepath.size());
  // /some/path/foobar.xdmf -> foobar.h5
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
  // /some/path/foobar.xdmf -> /some/path/
  String const h5filepath = filepath.substr(0, h5filepath_end);
  String const h5fullpath = h5filepath + h5filename;
  H5::H5File h5file(h5fullpath.data(), H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Add a uniform grid
  String const h5path;
  String const name = h5filename.substr(0, h5filename.size() - 3);
  writeXDMFUniformGrid(name, xdomain, h5file, h5filename, h5path, soup);
  // Write the XML file
  xdoc.save_file(filepath.data(), "  ");

  // Close the HDF5 file
  h5file.close();
} // writeXDMFFile

template <std::floating_point T>
void
addNodesToSoup(PolytopeSoup & mesh, Int const num_verts, Int const num_dimensions,
               H5::DataSet const & dataset, H5::FloatType const & datatype,
               bool const xyz)
{
  Vector<T> data_vec(num_verts * num_dimensions);
  dataset.read(data_vec.data(), datatype);
  // Add the nodes to the mesh
  mesh.reserveMoreVertices(num_verts);
  if (xyz) {
    for (Int i = 0; i < num_verts; ++i) {
      auto const x = static_cast<Float>(data_vec[i * 3]);
      auto const y = static_cast<Float>(data_vec[i * 3 + 1]);
      auto const z = static_cast<Float>(data_vec[i * 3 + 2]);
      mesh.addVertex(x, y, z);
    }
  } else { // XY
    for (Int i = 0; i < num_verts; ++i) {
      auto const x = static_cast<Float>(data_vec[i * 2]);
      auto const y = static_cast<Float>(data_vec[i * 2 + 1]);
      mesh.addVertex(x, y);
    }
  }
} // addNodesToSoup

void
readXDMFGeometry(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                 String const & h5filename, PolytopeSoup & soup)
{
  pugi::xml_node const xgeometry = xgrid.child("Geometry");
  if (strcmp(xgeometry.name(), "Geometry") != 0) {
    logger::error("XDMF geometry node not found");
    return;
  }
  // Get the geometry type
  String const geometry_type(xgeometry.attribute("GeometryType").value());
  if (geometry_type != "XYZ" && geometry_type != "XY") {
    logger::error("XDMF geometry type not supported: ", geometry_type);
    return;
  }
  // Get the DataItem node
  pugi::xml_node const xdataitem = xgeometry.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    logger::error("XDMF geometry DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Float") {
    logger::error("XDMF geometry data type not supported: ", data_type);
    return;
  }
  // Get the precision
  String const precision(xdataitem.attribute("Precision").value());
  if (precision != "4" && precision != "8") {
    logger::error("XDMF geometry precision not supported: ", precision);
    return;
  }
  // Get the dimensions
  String const dimensions(xdataitem.attribute("Dimensions").value());
  Int const split = dimensions.find_last_of(' ');
  char * end = nullptr;
  Int const num_verts = strto<Int>(dimensions.substr(0, split).data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  Int const num_dimensions = strto<Int>(dimensions.substr(split + 1).data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  if (geometry_type == "XYZ" && num_dimensions != 3) {
    logger::error("XDMF geometry dimensions not supported: ", dimensions);
    return;
  }
  if (geometry_type == "XY" && num_dimensions != 2) {
    logger::error("XDMF geometry dimensions not supported: ", dimensions);
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    logger::error("XDMF geometry format not supported: ", format);
    return;
  }

  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset =
      h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).data());
#  if UM2_ENABLE_ASSERTS
  H5T_class_t const type_class = dataset.getTypeClass();
  ASSERT(type_class == H5T_FLOAT);
#  endif
  H5::FloatType const datatype = dataset.getFloatType();
  size_t const datatype_size = datatype.getSize();
#  if UM2_ENABLE_ASSERTS
  ASSERT(datatype_size == strto<size_t>(precision.data(), &end));
  ASSERT(end != nullptr);
  end = nullptr;
  H5::DataSpace const dataspace = dataset.getSpace();
  int const rank = dataspace.getSimpleExtentNdims();
  ASSERT(rank == 2);
  hsize_t dims[2];
  int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
  ASSERT(ndims == 2);
  ASSERT(dims[0] == static_cast<hsize_t>(num_verts));
  ASSERT(dims[1] == static_cast<hsize_t>(num_dimensions));
#  endif
  if (datatype_size == 4) {
    addNodesToSoup<float>(soup, num_verts, num_dimensions, dataset, datatype,
                          geometry_type == "XYZ");
  } else if (datatype_size == 8) {
    addNodesToSoup<double>(soup, num_verts, num_dimensions, dataset, datatype,
                           geometry_type == "XYZ");
  }
}

template <std::signed_integral T>
void
addElementsToSoup(Int const num_elements, String const & topology_type,
                  String const & dimensions, PolytopeSoup & soup,
                  H5::DataSet const & dataset, H5::IntType const & datatype)
{
  if (topology_type == "Mixed") {
    // Expect dims to be one number
    char * end = nullptr;
    auto const conn_length = strto<Int>(dimensions.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    Vector<T> data_vec(conn_length);
    dataset.read(data_vec.data(), datatype);
    // Add the elements to the soup
    Int position = 0;
    Vector<Int> conn;
    for (Int i = 0; i < num_elements; ++i) {
      auto const element_type = static_cast<int8_t>(data_vec[position]);
      VTKElemType const elem_type = xdmfToVTKElemType(element_type);
      auto const npoints = verticesPerElem(elem_type);
      conn.resize(npoints);
      for (Int j = 0; j < npoints; ++j) {
        conn[j] = static_cast<Int>(data_vec[position + j + 1]);
      }
      position += npoints + 1;
      soup.addElement(elem_type, conn);
    }
  } else {
    Int const split = dimensions.find_last_of(' ');
    char * end = nullptr;
    auto const ncells = strto<Int>(dimensions.substr(0, split).data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    auto const nverts = strto<Int>(dimensions.substr(split + 1).data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;
    if (ncells != num_elements) {
      logger::error("Mismatch in number of elements");
      return;
    }
    Vector<T> data_vec(ncells * nverts);
    dataset.read(data_vec.data(), datatype);
    VTKElemType elem_type = VTKElemType::Invalid;
    if (topology_type == "Triangle") {
      elem_type = VTKElemType::Triangle;
    } else if (topology_type == "Quadrilateral") {
      elem_type = VTKElemType::Quad;
    } else if (topology_type == "Triangle_6") {
      elem_type = VTKElemType::QuadraticTriangle;
    } else if (topology_type == "Quadrilateral_8") {
      elem_type = VTKElemType::QuadraticQuad;
    } else {
      logger::error("Unsupported element type");
      return;
    }
    Vector<Int> conn(nverts);
    // Add the elements to the soup
    soup.reserveMoreElements(elem_type, ncells);
    for (Int i = 0; i < ncells; ++i) {
      for (Int j = 0; j < nverts; ++j) {
        conn[j] = static_cast<Int>(data_vec[i * nverts + j]);
      }
      soup.addElement(elem_type, conn);
    }
  }
}

void
readXDMFTopology(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                 String const & h5filename, PolytopeSoup & soup)
{
  pugi::xml_node const xtopology = xgrid.child("Topology");
  if (strcmp(xtopology.name(), "Topology") != 0) {
    logger::error("XDMF topology node not found");
    return;
  }
  // Get the topology type
  String const topology_type(xtopology.attribute("TopologyType").value());
  // Get the number of elements
  char * end = nullptr;
  Int const num_elements =
      strto<Int>(xtopology.attribute("NumberOfElements").value(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  // Get the DataItem node
  pugi::xml_node const xdataitem = xtopology.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    logger::error("XDMF topology DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Int") {
    logger::error("XDMF topology data type not supported: ", data_type);
    return;
  }
  // Get the precision
  String const precision(xdataitem.attribute("Precision").value());
  if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
    logger::error("XDMF topology precision not supported: ", precision);
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    logger::error("XDMF geometry format not supported: ", format);
    return;
  }
  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset =
      h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).data());
#  if UM2_ENABLE_ASSERTS
  H5T_class_t const type_class = dataset.getTypeClass();
  ASSERT(type_class == H5T_INTEGER);
#  endif
  H5::IntType const datatype = dataset.getIntType();
  size_t const datatype_size = datatype.getSize();
#  if UM2_ENABLE_ASSERTS
  ASSERT(datatype_size == strto<size_t>(precision.data(), &end));
  ASSERT(end != nullptr);
  end = nullptr;
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
#  endif
  // Get the dimensions
  String const dimensions(xdataitem.attribute("Dimensions").value());
  if (datatype_size == 4) {
    addElementsToSoup<int32_t>(num_elements, topology_type, dimensions, soup, dataset,
                               datatype);
  } else if (datatype_size == 8) {
    addElementsToSoup<int64_t>(num_elements, topology_type, dimensions, soup, dataset,
                               datatype);
  } else {
    logger::error("Unsupported data type size");
    return;
  }
}

//==============================================================================
// addElsetToSoup
//==============================================================================

template <std::signed_integral T, std::floating_point U>
void
addElsetToSoup(PolytopeSoup & soup, Int const num_elements, H5::DataSet const & dataset,
               H5::IntType const & datatype, String const & elset_name,
               bool const has_attribute, H5::DataSet const & attribute_dataset,
               H5::FloatType const & attribute_datatype)
{
  Vector<T> data_vec(num_elements);
  dataset.read(data_vec.data(), datatype);
  Vector<Int> elset_ids(num_elements);
  for (Int i = 0; i < num_elements; ++i) {
    elset_ids[i] = static_cast<Int>(data_vec[i]);
  }
  if (!has_attribute) {
    soup.addElset(elset_name, elset_ids);
    return;
  }
  Vector<U> attribute_data_vec;
  attribute_data_vec.resize(num_elements);
  attribute_dataset.read(attribute_data_vec.data(), attribute_datatype);
  Vector<Float> elset_data(num_elements);
  for (Int i = 0; i < num_elements; ++i) {
    elset_data[i] = static_cast<Float>(attribute_data_vec[i]);
  }
  soup.addElset(elset_name, elset_ids, elset_data);
}

//==============================================================================
// readXDMFElsets
//==============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
readXDMFElsets(pugi::xml_node const & xgrid, H5::H5File const & h5file,
               String const & h5filename, PolytopeSoup & soup)
{
  LOG_DEBUG("Reading XDMF elsets");
  // Loop over all nodes to find the elsets
  for (pugi::xml_node xelset = xgrid.first_child(); xelset != nullptr;
       xelset = xelset.next_sibling()) {
    if (strcmp(xelset.name(), "Set") != 0) {
      continue;
    }
    // Get the SetType
    String const set_type(xelset.attribute("SetType").value());
    if (set_type != "Cell") {
      logger::error("XDMF elset only supports SetType=Cell");
      return;
    }
    // Get the name
    String const name(xelset.attribute("Name").value());
    if (name.empty()) {
      logger::error("XDMF elset name not found");
      return;
    }
    // Get the DataItem node
    pugi::xml_node const xdataitem = xelset.child("DataItem");
    if (strcmp(xdataitem.name(), "DataItem") != 0) {
      logger::error("XDMF elset DataItem node not found");
      return;
    }
    // Get the data type
    String const data_type(xdataitem.attribute("DataType").value());
    if (data_type != "Int") {
      logger::error("XDMF elset data type not supported: ", data_type);
      return;
    }
    // Get the precision
    String const precision(xdataitem.attribute("Precision").value());
    if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
      logger::error("XDMF elset precision not supported: ", precision);
      return;
    }
    // Get the format
    String const format(xdataitem.attribute("Format").value());
    if (format != "HDF") {
      logger::error("XDMF elset format not supported: ", format);
      return;
    }
    // Get the h5 dataset path
    String const h5dataset(xdataitem.child_value());
    // Read the data
    H5::DataSet const dataset =
        h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).data());
#  if UM2_ENABLE_ASSERTS
    H5T_class_t const type_class = dataset.getTypeClass();
    ASSERT(type_class == H5T_INTEGER);
    char * end = nullptr;
#  endif
    H5::IntType const datatype = dataset.getIntType();
    size_t const datatype_size = datatype.getSize();
    ASSERT(datatype_size == strto<size_t>(precision.data(), &end));
    ASSERT(end != nullptr);
    H5::DataSpace const dataspace = dataset.getSpace();
#  if UM2_ENABLE_ASSERTS
    end = nullptr;
    int const rank = dataspace.getSimpleExtentNdims();
    ASSERT(rank == 1);
#  endif

    hsize_t dims[1];
#  if UM2_ENABLE_ASSERTS
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    ASSERT(ndims == 1);
    String const dimensions(xdataitem.attribute("Dimensions").value());
#  else
    dataspace.getSimpleExtentDims(dims, nullptr);
#  endif
    auto const num_elements = static_cast<Int>(dims[0]);
    ASSERT(num_elements == strto<Int>(dimensions.data(), &end));
    ASSERT(end != nullptr);
    //    end = nullptr;

    // Check if there is an associated data set
    // Get the Attribute node
    bool has_attribute = false;
    size_t att_datatype_size = sizeof(Float);
    H5::DataSet att_dataset;
    H5::FloatType att_datatype;
    pugi::xml_node const xattribute = xelset.child("Attribute");
    if (strcmp(xattribute.name(), "Attribute") == 0) {
      LOG_DEBUG("Found data associated with elset: ", name);
      has_attribute = true;

      // Ensure Center="Cell"
      String const center(xattribute.attribute("Center").value());
      if (center != "Cell") {
        logger::error("Only elset attribute data with Center=Cell is supported");
        return;
      }

      // Get the DataItem node
      pugi::xml_node const xattdataitem = xattribute.child("DataItem");
      if (strcmp(xattdataitem.name(), "DataItem") != 0) {
        logger::error("XDMF elset attribute DataItem node not found");
        return;
      }

      // Get the data type
      String const att_data_type(xattdataitem.attribute("DataType").value());
      if (att_data_type != "Float") {
        logger::error("XDMF elset attribute data type not supported: ", att_data_type);
        // has_attribute = false;
        return;
      }

      // Get the precision
      String const att_precision(xattdataitem.attribute("Precision").value());
      if (att_precision != "4" && att_precision != "8") {
        logger::error("XDMF elset attribute precision not supported: ", att_precision);
        // has_attribute = false;
        return;
      }

      // Get the format
      String const att_format(xattdataitem.attribute("Format").value());
      if (att_format != "HDF") {
        logger::error("XDMF elset attribute format not supported: ", att_format);
        // has_attribute = false;
        return;
      }

      // Get the h5 dataset path
      String const h5attdataset(xattdataitem.child_value());
      // Read the data
      att_dataset = h5file.openDataSet(h5attdataset.substr(h5filename.size() + 1).data());
#  if UM2_ENABLE_ASSERTS
      H5T_class_t const att_type_class = att_dataset.getTypeClass();
      ASSERT(att_type_class == H5T_FLOAT);
      char * att_end = nullptr;
#  endif
      att_datatype = att_dataset.getFloatType();
      att_datatype_size = att_datatype.getSize();
      ASSERT(att_datatype_size == strto<size_t>(att_precision.data(), &att_end));
      ASSERT(att_end != nullptr);
      H5::DataSpace const att_dataspace = att_dataset.getSpace();
#  if UM2_ENABLE_ASSERTS
      att_end = nullptr;
      int const att_rank = att_dataspace.getSimpleExtentNdims();
      ASSERT(att_rank == 1);
#  endif

      hsize_t att_dims[1];
#  if UM2_ENABLE_ASSERTS
      int const att_ndims = att_dataspace.getSimpleExtentDims(att_dims, nullptr);
      ASSERT(att_ndims == 1);
      String const att_dimensions(xattdataitem.attribute("Dimensions").value());
      auto const att_num_elements = static_cast<Int>(dims[0]);
#  else
      att_dataspace.getSimpleExtentDims(att_dims, nullptr);
#  endif
      ASSERT(att_num_elements == strto<Int>(att_dimensions.data(), &att_end));
      ASSERT(att_end != nullptr);
      //      att_end = nullptr;
      ASSERT(att_num_elements == num_elements);
    } // if (strcmp(xattribute.name(), "Attribute") == 0)

    LOG_DEBUG("has_attribute: ", has_attribute);

    if (datatype_size == 4) {
      if (att_datatype_size == 4) {
        addElsetToSoup<int32_t, float>(soup, num_elements, dataset, datatype, name,
                                       has_attribute, att_dataset, att_datatype);
      } else if (att_datatype_size == 8) {
        addElsetToSoup<int32_t, double>(soup, num_elements, dataset, datatype, name,
                                        has_attribute, att_dataset, att_datatype);
      } else {
        logger::error("Unsupported attribute data type size");
      }
    } else if (datatype_size == 8) {
      if (att_datatype_size == 4) {
        addElsetToSoup<int64_t, float>(soup, num_elements, dataset, datatype, name,
                                       has_attribute, att_dataset, att_datatype);
      } else if (att_datatype_size == 8) {
        addElsetToSoup<int64_t, double>(soup, num_elements, dataset, datatype, name,
                                        has_attribute, att_dataset, att_datatype);
      } else {
        logger::error("Unsupported attribute data type size");
      }
    }
  }
}

} // namespace

//==============================================================================
// readXDMFUniformGrid
//==============================================================================

void
readXDMFUniformGrid(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                    String const & h5filename, PolytopeSoup & soup)
{
  readXDMFGeometry(xgrid, h5file, h5filename, soup);
  readXDMFTopology(xgrid, h5file, h5filename, soup);
  readXDMFElsets(xgrid, h5file, h5filename, soup);
}

//==============================================================================
// readXDMFFile
//==============================================================================

namespace
{

void
readXDMFFile(String const & filename, PolytopeSoup & soup)
{
  logger::info("Reading XDMF file: " + filename);

  // Open HDF5 file
  Int last_slash = filename.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = 0;
  }
  Int const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
  ASSERT(h5filepath_end < filename.size());
  String const h5filename =
      filename.substr(h5filepath_end, filename.size() - 5 - h5filepath_end) + ".h5";
  String const h5filepath = filename.substr(0, h5filepath_end);
  String const h5fullpath = h5filepath + h5filename;
  H5::H5File h5file(h5fullpath.data(), H5F_ACC_RDONLY);

  // Setup XML file
  pugi::xml_document xdoc;
  pugi::xml_parse_result const result = xdoc.load_file(filename.data());
  if (!result) {
    logger::error("XDMF XML parse error: ", result.description(),
                  ", character pos= ", result.offset);
    return;
  }
  pugi::xml_node const xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    logger::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node const xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    logger::error("XDMF XML domain node is not Domain");
    return;
  }

  pugi::xml_node const xgrid = xdomain.child("Grid");
  if (strcmp("Grid", xgrid.name()) != 0) {
    logger::error("XDMF XML grid node is not Grid");
    return;
  }
  if (strcmp("Uniform", xgrid.attribute("GridType").value()) == 0) {
    readXDMFUniformGrid(xgrid, h5file, h5filename, soup);
  } else if (strcmp("Tree", xgrid.attribute("GridType").value()) == 0) {
    logger::error("XDMF XML Tree is not supported");
  } else {
    logger::error("XDMF XML grid type is not Uniform or Tree");
  }
  // Close HDF5 file
  h5file.close();
  // Close XML file
  xdoc.reset();
  logger::info("Finished reading XDMF file: ", filename);
}

} // namespace

#endif // UM2_HAS_XDMF

//==============================================================================
// IO
//==============================================================================

void
PolytopeSoup::read(String const & filename)
{
  if (filename.ends_with(".inp")) {
    readAbaqusFile(filename, *this);
  } else if (filename.ends_with(".vtk")) {
    readVTKFile(filename, *this);
#if UM2_HAS_XDMF
  } else if (filename.ends_with(".xdmf")) {
    readXDMFFile(filename, *this);
#endif
  } else {
    logger::error("Unsupported file format.");
  }
}

#if UM2_HAS_XDMF
void
PolytopeSoup::write(String const & filename) const
{
  if (filename.ends_with(".xdmf")) {
    writeXDMFFile(filename, *this);
  } else {
    logger::error("Unsupported file format.");
  }
}
#endif

} // namespace um2

// NOLINTEND(misc-include-cleaner)
