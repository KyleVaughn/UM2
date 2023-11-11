#pragma once

#include <um2/common/log.hpp>
#include <um2/geometry/point.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/memory.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

#include <vector>

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
    ASSERT(false);
    return -1;
  }
}

constexpr auto
verticesPerElem(MeshType const type) -> Size
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
    ASSERT(false);
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
    ASSERT(false);
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
    ASSERT(false);
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
  // more efficient for generating submeshes and other time-critical operations.
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

  void
  getSubmesh(String const & elset_name, PolytopeSoup<T, I> & submesh) const;
  
  void
  getMaterialIDs(Vector<MaterialID> & material_ids,
                 Vector<String> const & material_names) const;
  
}; // struct PolytopeSoup

//==============================================================================
// numElems
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::numElems() const -> Size
{
  return element_types.size();
}

//==============================================================================
// hasElsetData
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::hasElsetData() const -> bool
{
  return std::ranges::any_of(elset_data.cbegin(), elset_data.cend(),
                             [](auto const & data) { return !data.empty(); });
}

//==============================================================================
// getElemTypes
//==============================================================================
// Get all the element types in the mesh.

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
PolytopeSoup<T, I>::getElemTypes() const -> Vec<8, VTKElemType>
{
  Vec<8, VTKElemType> el_types;
  um2::fill(el_types.begin(), el_types.end(), VTKElemType::None);
  for (auto const & this_type : element_types) {
    bool found = false;
    for (Size i = 0; i < 8; ++i) {
      if (this_type == el_types[i]) {
        found = true;
        break;
      }
    }
    if (!found) {
      for (Size i = 0; i < 8; ++i) {
        if (el_types[i] == VTKElemType::None) {
          el_types[i] = this_type;
          break;
        }
      }
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
  for (auto const & this_type : element_types) {
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
// compareGeometry
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(PolytopeSoup<T, I> const & lhs, PolytopeSoup<T, I> const & rhs) -> int
{
  if (lhs.vertices.size() != rhs.vertices.size()) {
    return 1;
  }
  auto const compare = [](Point3<T> const & a, Point3<T> const & b) -> bool {
    return um2::isApprox(a, b);
  };
  if (!std::equal(lhs.vertices.cbegin(), lhs.vertices.cend(), rhs.vertices.cbegin(),
                  compare)) {
    return 2;
  }
  return 0;
}

//==============================================================================
// compareTopology
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareTopology(PolytopeSoup<T, I> const & lhs, PolytopeSoup<T, I> const & rhs) -> int
{
  if (lhs.element_types.size() != rhs.element_types.size()) {
    return 1;
  }
  if (!std::equal(lhs.element_types.cbegin(), lhs.element_types.cend(),
                  rhs.element_types.cbegin())) {
    return 2;
  }
  if (!std::equal(lhs.element_conn.cbegin(), lhs.element_conn.cend(),
                  rhs.element_conn.cbegin())) {
    return 3;
  }

  // If the element types and connectivity are the same, then the element
  // offsets SHOULD be the same.
#ifndef NDEBUG
  if (lhs.element_offsets.size() != rhs.element_offsets.size()) {
    return 4;
  }
  if (!std::equal(lhs.element_offsets.cbegin(), lhs.element_offsets.cend(),
                  rhs.element_offsets.cbegin())) {
    return 5;
  }
#endif
  return 0;
}

//==============================================================================
// addElset
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr void
PolytopeSoup<T, I>::addElset(String const & name, Vector<I> const & ids, Vector<T> data)
{
  LOG_DEBUG("Adding elset: " + name);

  for (auto const & this_name : elset_names) {
    // cppcheck-suppress useStlAlgorithm; justification: This is more clear.
    if (this_name == name) {
      LOG_ERROR("Elset " + name + " already exists.");
      return;
    }
  }

  Size const num_ids = ids.size();
  if (num_ids == 0) {
    LOG_ERROR("Elset ids" + name + " is empty.");
    return;
  }

  if (!data.empty() && (data.size() != num_ids)) {
    LOG_ERROR("Elset data size does not match the number of ids.");
    return;
  }

  elset_names.emplace_back(name);
  if (elset_offsets.empty()) {
    elset_offsets.push_back(0);
  }

  Size const old_num_ids = elset_ids.size();
  Size const new_num_ids = old_num_ids + num_ids;
  elset_offsets.push_back(static_cast<I>(new_num_ids));
  elset_ids.resize(new_num_ids);
  um2::copy(ids.begin(), ids.end(), elset_ids.data() + old_num_ids);
  elset_data.emplace_back(um2::move(data));
}

//==============================================================================
// sortElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr void
PolytopeSoup<T, I>::sortElsets()
{
  using NameOffsetsPair = std::pair<String, std::pair<I, I>>;
  // Create a vector containing the elset names and offsets.
  Size const num_elsets = elset_names.size();
  Vector<NameOffsetsPair> elset_name_offsets_pairs(num_elsets);
  for (Size i = 0; i < num_elsets; ++i) {
    elset_name_offsets_pairs[i] = std::make_pair(
        elset_names[i], std::make_pair(elset_offsets[i], elset_offsets[i + 1]));
  }
  // Sort the vector by the elset names.
  // This is only of length num_elsets, so it should be fast. No need to
  // parallelize.
  std::sort(elset_name_offsets_pairs.begin(), elset_name_offsets_pairs.end(),
            [](NameOffsetsPair const & a, NameOffsetsPair const & b) -> bool {
              return a.first < b.first;
            });
  // Create a vector to store the sorted elset ids.
  Vector<I> elset_ids_copy = elset_ids;
  auto const * const elset_ids_copy_ptr = elset_ids_copy.data();
  auto * const elset_ids_ptr = elset_ids.data();
  // Overwrite the current elset offsets and
  // copy the sorted elset ids to the elset_ids_copy vector.
  I offset = 0;
  for (Size i = 0; i < num_elsets; ++i) {
    elset_names[i] = elset_name_offsets_pairs[i].first;
    auto const & offset_pair = elset_name_offsets_pairs[i].second;
    I const len = offset_pair.second - offset_pair.first;
    elset_offsets[i] = offset;
    elset_offsets[i + 1] = offset + len;
    copy(elset_ids_copy_ptr + offset_pair.first, elset_ids_copy_ptr + offset_pair.second,
         elset_ids_ptr + offset);
    offset += len;
  }
  if (hasElsetData()) {
    Log::warn("PolytopeSoup.sortElsets: elset data is not currently sorted");
  }
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
  for (auto const & elset_name : elset_names) {
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
  auto const * const elset_it =
      std::find(elset_names.cbegin(), elset_names.cend(), elset_name);
  if (elset_it == elset_names.cend()) {
    Log::error("getSubmesh: Elset '" + elset_name + "' not found");
    return;
  }

  // Get the element ids in the elset.
  auto const elset_index = static_cast<Size>(elset_it - elset_names.cbegin());
  auto const submesh_elset_start = static_cast<Size>(elset_offsets[elset_index]);
  auto const submesh_elset_end = static_cast<Size>(elset_offsets[elset_index + 1]);
  auto const submesh_num_elements = submesh_elset_end - submesh_elset_start;
  Vector<I> element_ids(submesh_num_elements);
  for (Size i = 0; i < submesh_num_elements; ++i) {
    element_ids[i] = elset_ids[submesh_elset_start + i];
  }
#if UM2_USE_TBB
  std::sort(std::execution::par_unseq, element_ids.begin(), element_ids.end());
#else
  std::sort(element_ids.begin(), element_ids.end());
#endif

  // Get the element connectivity and remap the vertex ids.
  submesh.element_types.resize(submesh_num_elements);
  submesh.element_offsets.resize(submesh_num_elements + 1);
  submesh.element_offsets[0] = 0;
  submesh.element_conn.reserve(3 * submesh_num_elements);
  // push_back creates race condition. Don't parallelize.
  for (Size i = 0; i < submesh_num_elements; ++i) {
    auto const element_id = static_cast<Size>(element_ids[i]);
    submesh.element_types[i] = element_types[element_id];
    auto const element_start = static_cast<Size>(element_offsets[element_id]);
    auto const element_end = static_cast<Size>(element_offsets[element_id + 1]);
    auto const element_len = element_end - element_start;
    submesh.element_offsets[i + 1] =
        submesh.element_offsets[i] + static_cast<I>(element_len);
    for (Size j = 0; j < element_len; ++j) {
      I const vertex_id = element_conn[element_start + j];
      submesh.element_conn.push_back(vertex_id);
    }
  }
  // Get the unique vertex ids.
  Vector<I> unique_vertex_ids = submesh.element_conn;
#if UM2_USE_TBB
  std::sort(std::execution::par_unseq, unique_vertex_ids.begin(),
            unique_vertex_ids.end());
  auto * const last = std::unique(std::execution::par_unseq, unique_vertex_ids.begin(),
                                  unique_vertex_ids.end());
#else
  std::sort(unique_vertex_ids.begin(), unique_vertex_ids.end());
  auto * const last = std::unique(unique_vertex_ids.begin(), unique_vertex_ids.end());
#endif
  auto const num_unique_verts = static_cast<Size>(last - unique_vertex_ids.cbegin());
  // We now have the unique vertex ids. We need to remap the connectivity.
  // unique_vertex_ids[i] is the old vertex id, and i is the new vertex id.
#if UM2_USE_OPENMP
#  pragma omp parallel for
#endif
  for (Size i = 0; i < submesh.element_conn.size(); ++i) {
    I const old_vertex_id = submesh.element_conn[i];
    auto * const it = std::lower_bound(unique_vertex_ids.begin(), last, old_vertex_id);
    auto const new_vertex_id = static_cast<I>(it - unique_vertex_ids.cbegin());
    ASSERT(*it == old_vertex_id);
    submesh.element_conn[i] = new_vertex_id;
  }

  // Get the x, y, z coordinates for the vertices.
  submesh.vertices.resize(num_unique_verts);
  for (Size i = 0; i < num_unique_verts; ++i) {
    auto const vertex_id = static_cast<Size>(unique_vertex_ids[i]);
    submesh.vertices[i] = vertices[vertex_id];
  }

  Size const num_elsets = elset_names.size();
  // If the intersection of this elset and another elset is non-empty, then we need to
  // add the itersection as an elset and remap the elset IDs using the element_ids
  // vector.
  // element_ids[i] is the old element id, and i is the new element id.
  //
  // push_back causes race condition. Don't parallelize.
  for (Size i = 0; i < num_elsets; ++i) {
    if (i == elset_index) {
      continue;
    }
    auto const elset_start = static_cast<Size>(elset_offsets[i]);
    auto const elset_end = static_cast<Size>(elset_offsets[i + 1]);
    auto * const elset_ids_begin = addressof(elset_ids[elset_start]);
    auto * const elset_ids_end = elset_ids_begin + (elset_end - elset_start);
    std::vector<I> intersection;
    std::set_intersection(element_ids.begin(), element_ids.end(), elset_ids_begin,
                          elset_ids_end, std::back_inserter(intersection));
    if (intersection.empty()) {
      continue;
    }
    // We have an intersection. Add the elset.
    submesh.elset_names.push_back(elset_names[i]);
    if (submesh.elset_offsets.empty()) {
      submesh.elset_offsets.push_back(0);
    }
    submesh.elset_offsets.push_back(submesh.elset_offsets.back() +
                                    static_cast<I>(intersection.size()));
    for (size_t j = 0; j < intersection.size(); ++j) {
      I const old_element_id = intersection[j];
      auto * const it =
          std::lower_bound(element_ids.begin(), element_ids.end(), old_element_id);
      submesh.elset_ids.push_back(static_cast<I>(it - element_ids.begin()));
    }
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
    for (Size j = 0; j < elset_names.size(); ++j) {
      if (elset_names[j] == mat_name) {
        auto const start = static_cast<Size>(this->elset_offsets[j]);
        auto const end = static_cast<Size>(this->elset_offsets[j + 1]);
        for (Size k = start; k < end; ++k) {
          auto const elem = static_cast<Size>(this->elset_ids[k]);
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

} // namespace um2