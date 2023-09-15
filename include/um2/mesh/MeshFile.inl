namespace um2
{

//==============================================================================
// numCells
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
MeshFile<T, I>::numCells() const -> size_t
{
  return element_offsets.size() - 1;
}

//==============================================================================
// getMeshType
//==============================================================================

template <std::floating_point T, std::signed_integral I>
PURE constexpr auto
MeshFile<T, I>::getMeshType() const -> MeshType
{
  // Loop throught the element types to determine which 1 or 2 mesh types are
  // present.
  MeshType type1 = MeshType::None;
  MeshType type2 = MeshType::None;
  for (auto const & this_type : element_types) {
    if (type1 == MeshType::None) {
      type1 = this_type;
    }
    if (type1 == this_type) {
      continue;
    }
    if (type2 == MeshType::None) {
      type2 = this_type;
    }
    if (type2 == this_type) {
      continue;
    }
    return MeshType::None;
  }
  // Determine the mesh type from the 1 or 2 mesh types.
  if (type1 == MeshType::Tri && type2 == MeshType::None) {
    return MeshType::Tri;
  }
  if (type1 == MeshType::Quad && type2 == MeshType::None) {
    return MeshType::Quad;
  }
  if ((type1 == MeshType::Tri && type2 == MeshType::Quad) ||
      (type1 == MeshType::Quad && type2 == MeshType::Tri)) {
    return MeshType::TriQuad;
  }
  if (type1 == MeshType::QuadraticTri && type2 == MeshType::None) {
    return MeshType::QuadraticTri;
  }
  if (type1 == MeshType::QuadraticQuad && type2 == MeshType::None) {
    return MeshType::QuadraticQuad;
  }
  if ((type1 == MeshType::QuadraticTri && type2 == MeshType::QuadraticQuad) ||
      (type1 == MeshType::QuadraticQuad && type2 == MeshType::QuadraticTri)) {
    return MeshType::QuadraticTriQuad;
  }
  return MeshType::None;
}

//==============================================================================
// compareGeometry
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(MeshFile<T, I> const & lhs, MeshFile<T, I> const & rhs) -> int
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
compareTopology(MeshFile<T, I> const & lhs, MeshFile<T, I> const & rhs) -> int
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
// sortElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr void
MeshFile<T, I>::sortElsets()
{
  using NameOffsetsPair = std::pair<std::string, std::pair<I, I>>;
  // Create a vector containing the elset names and offsets.
  size_t const num_elsets = elset_names.size();
  std::vector<NameOffsetsPair> elset_name_offsets_pairs(num_elsets);
  for (size_t i = 0; i < num_elsets; ++i) {
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
  std::vector<I> elset_ids_copy = elset_ids;
  // Overwrite the current elset offsets and
  // copy the sorted elset ids to the elset_ids_copy vector.
  I offset = 0;
  for (size_t i = 0; i < num_elsets; ++i) {
    elset_names[i] = elset_name_offsets_pairs[i].first;
    auto const & offset_pair = elset_name_offsets_pairs[i].second;
    I const len = offset_pair.second - offset_pair.first;
    elset_offsets[i] = offset;
    elset_offsets[i + 1] = offset + len;
    copy(addressof(elset_ids_copy[static_cast<size_t>(offset_pair.first)]),
         addressof(elset_ids_copy[static_cast<size_t>(offset_pair.second)]),
         addressof(elset_ids[static_cast<size_t>(offset)]));
    offset += len;
  }
}

//==============================================================================
// getSubmesh
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
MeshFile<T, I>::getSubmesh(std::string const & elset_name, MeshFile<T, I> & submesh) const
{
  LOG_DEBUG("Extracting submesh for elset: " + elset_name);

  // Find the elset with the given name.
  auto const elset_it = std::find(elset_names.cbegin(), elset_names.cend(), elset_name);
  if (elset_it == elset_names.cend()) {
    Log::error("getSubmesh: Elset '" + elset_name + "' not found");
    return;
  }

  submesh.filepath = "";
  submesh.name = elset_name;
  submesh.format = format;

  // Get the element ids in the elset.
  auto const elset_index = static_cast<size_t>(elset_it - elset_names.cbegin());
  auto const submesh_elset_start = static_cast<size_t>(elset_offsets[elset_index]);
  auto const submesh_elset_end = static_cast<size_t>(elset_offsets[elset_index + 1]);
  auto const submesh_num_elements = submesh_elset_end - submesh_elset_start;
  std::vector<I> element_ids(submesh_num_elements);
  for (size_t i = 0; i < submesh_num_elements; ++i) {
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
  submesh.element_conn.reserve(3 *
                               submesh_num_elements); // 3 is the min vertices per element
  // push_back creates race condition. Don't parallelize.
  for (size_t i = 0; i < submesh_num_elements; ++i) {
    auto const element_id = static_cast<size_t>(element_ids[i]);
    submesh.element_types[i] = element_types[element_id];
    auto const element_start = static_cast<size_t>(element_offsets[element_id]);
    auto const element_end = static_cast<size_t>(element_offsets[element_id + 1]);
    auto const element_len = element_end - element_start;
    submesh.element_offsets[i + 1] =
        submesh.element_offsets[i] + static_cast<I>(element_len);
    for (size_t j = 0; j < element_len; ++j) {
      I const vertex_id = element_conn[element_start + j];
      submesh.element_conn.push_back(vertex_id);
    }
  }
  // Get the unique vertex ids.
  std::vector<I> all_vertex_ids = submesh.element_conn;
#if UM2_USE_TBB
  std::sort(std::execution::par_unseq, all_vertex_ids.begin(), all_vertex_ids.end());
  auto const last = std::unique(std::execution::par_unseq, all_vertex_ids.begin(),
                                all_vertex_ids.end());
#else
  std::sort(all_vertex_ids.begin(), all_vertex_ids.end());
  auto const last = std::unique(all_vertex_ids.begin(), all_vertex_ids.end());
#endif
  // This is an unnecessary copy
  std::vector<I> unique_vertex_ids(all_vertex_ids.begin(), last);
  // We now have the unique vertex ids. We need to remap the connectivity.
  // unique_vertex_ids[i] is the old vertex id, and i is the new vertex id.
#if UM2_USE_OPENMP
#  pragma omp parallel for
#endif
  for (size_t i = 0; i < submesh.element_conn.size(); ++i) {
    I const old_vertex_id = submesh.element_conn[i];
    auto const it = std::lower_bound(unique_vertex_ids.begin(), unique_vertex_ids.end(),
                                     old_vertex_id);
    auto const new_vertex_id = static_cast<I>(it - unique_vertex_ids.begin());
    assert(*it == old_vertex_id);
    submesh.element_conn[i] = new_vertex_id;
  }

  // Get the x, y, z coordinates for the vertices.
  submesh.vertices.resize(unique_vertex_ids.size());
  for (size_t i = 0; i < unique_vertex_ids.size(); ++i) {
    auto const vertex_id = static_cast<size_t>(unique_vertex_ids[i]);
    submesh.vertices[i] = vertices[vertex_id];
  }

  size_t const num_elsets = elset_names.size();
  // If the intersection of this elset and another elset is non-empty, then we need to
  // add the itersection as an elset and remap the elset IDs using the element_ids vector.
  // element_ids[i] is the old element id, and i is the new element id.
  //
  // push_back causes race condition. Don't parallelize.
  for (size_t i = 0; i < num_elsets; ++i) {
    if (i == elset_index) {
      continue;
    }
    auto const elset_start = static_cast<size_t>(elset_offsets[i]);
    auto const elset_end = static_cast<size_t>(elset_offsets[i + 1]);
    std::vector<I> intersection;
    std::set_intersection(
        element_ids.begin(), element_ids.end(), addressof(elset_ids[elset_start]),
        addressof(elset_ids[elset_end]), std::back_inserter(intersection));
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
      auto const it =
          std::lower_bound(element_ids.begin(), element_ids.end(), old_element_id);
      submesh.elset_ids.push_back(static_cast<I>(it - element_ids.begin()));
    }
  }
}

//==============================================================================
// getMaterialNames
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
MeshFile<T, I>::getMaterialNames(std::vector<std::string> & material_names) const
{
  std::string const material = "Material";
  for (auto const & elset_name : elset_names) {
    size_t const name_len = elset_name.size();
    if (name_len >= 10 && elset_name.starts_with(material)) {
      material_names.push_back(elset_name);
    }
  }
  // Should already be sorted
  assert(std::is_sorted(material_names.begin(), material_names.end()));
}

//==============================================================================
// getMaterialIDs
//==============================================================================

template <std::floating_point T, std::signed_integral I>
constexpr void
MeshFile<T, I>::getMaterialIDs(std::vector<MaterialID> & material_ids,
                               std::vector<std::string> const & material_names) const
{
  material_ids.resize(numCells(), static_cast<MaterialID>(-1));
  size_t const nmats = material_names.size();
  for (size_t i = 0; i < nmats; ++i) {
    std::string const & mat_name = material_names[i];
    for (size_t j = 0; j < elset_names.size(); ++j) {
      if (elset_names[j] == mat_name) {
        auto const start = static_cast<size_t>(this->elset_offsets[j]);
        auto const end = static_cast<size_t>(this->elset_offsets[j + 1]);
        for (size_t k = start; k < end; ++k) {
          auto const elem = static_cast<size_t>(this->elset_ids[k]);
          if (material_ids[elem] != -1) {
            Log::error("Element " + std::to_string(elem) + " has multiple materials");
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
