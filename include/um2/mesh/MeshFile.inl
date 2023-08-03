namespace um2
{

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareGeometry(MeshFile<T, I> const & a, MeshFile<T, I> const & b) -> int
{
  auto compare_floats = [](T const x, T const y) -> bool {
    return um2::abs(x - y) < epsilonDistance<T>();
  };

  if (a.nodes_x.size() != b.nodes_x.size() ||
      !std::equal(a.nodes_x.cbegin(), a.nodes_x.cend(), b.nodes_x.cbegin(),
                  compare_floats)) {
    return 1;
  }
  if (a.nodes_y.size() != b.nodes_y.size() ||
      !std::equal(a.nodes_y.cbegin(), a.nodes_y.cend(), b.nodes_y.cbegin(),
                  compare_floats)) {
    return 2;
  }
  if (a.nodes_z.size() != b.nodes_z.size() ||
      !std::equal(a.nodes_z.cbegin(), a.nodes_z.cend(), b.nodes_z.cbegin(),
                  compare_floats)) {
    return 3;
  }
  return 0;
}

template <std::floating_point T, std::signed_integral I>
constexpr auto
compareTopology(MeshFile<T, I> const & a, MeshFile<T, I> const & b) -> int
{
  if (a.element_offsets.size() != b.element_offsets.size() ||
      !std::equal(a.element_offsets.cbegin(), a.element_offsets.cend(),
                  b.element_offsets.cbegin())) {
    return 1;
  }
  if (a.element_conn.size() != b.element_conn.size() ||
      !std::equal(a.element_conn.cbegin(), a.element_conn.cend(),
                  b.element_conn.cbegin())) {
    return 2;
  }
  return 0;
}

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

template <std::floating_point T, std::signed_integral I>
constexpr void MeshFile<T, I>::getSubmesh(std::string const & elset_name,
                                           MeshFile<T, I> & submesh) const
{
  Log::debug("Extracting submesh for elset: " + elset_name);
  // Find the elset with the given name.
  auto const it = std::find(elset_names.cbegin(), elset_names.cend(), elset_name);
  if (it == elset_names.cend()) {
      Log::error("Elset not found");
      return;
  }

  submesh.filepath = "";
  submesh.name = elset_name;
  submesh.format = format;

  // Get the element ids in the elset.
  auto const elset_index = static_cast<size_t>(it - elset_names.cbegin());
  auto const submesh_elset_start = static_cast<size_t>(elset_offsets[elset_index]);
  auto const submesh_elset_end   = static_cast<size_t>(elset_offsets[elset_index + 1]);
  auto const submesh_num_elements = submesh_elset_end - submesh_elset_start; 
  std::vector<I> element_ids(submesh_num_elements); 
  for (size_t i = 0; i < submesh_num_elements; ++i) {
    element_ids[i] = elset_ids[submesh_elset_start + i];
  }
  std::sort(element_ids.begin(), element_ids.end());

  // Get the element types, offsets, and connectivity. We will also get the unique node ids,
  // since we need to remap the connectivity.
  std::vector<I> unique_node_ids;
  submesh.element_types.resize(submesh_num_elements);
  submesh.element_offsets.resize(submesh_num_elements + 1);
  submesh.element_offsets[0] = 0;
  for (size_t i = 0; i < submesh_num_elements; ++i) {
    auto const element_id = static_cast<size_t>(element_ids[i]);
    submesh.element_types[i] = element_types[element_id];
    auto const element_start = static_cast<size_t>(element_offsets[element_id]);
    auto const element_end   = static_cast<size_t>(element_offsets[element_id + 1]);
    auto const element_len   = element_end - element_start;
    submesh.element_offsets[i + 1] = submesh.element_offsets[i] + static_cast<I>(element_len);
    for (size_t j = 0; j < element_len; ++j) {
        I const node_id = element_conn[element_start + j]; 
        submesh.element_conn.push_back(node_id); 
        auto const it2 = std::lower_bound(unique_node_ids.cbegin(),
                                         unique_node_ids.cend(), node_id);
        if (it2 == unique_node_ids.cend() || *it2 != node_id) {
            unique_node_ids.insert(it2, node_id);
        }
    }
  }
 // We now have the unique node ids. We need to remap the connectivity.
 // unique_node_ids[i] is the old node id, and i is the new node id.
 for (length_t i = 0; i < submesh.element_conn.size(); ++i) {
     I const old_node_id = submesh.element_conn[i];
     auto const it = std::lower_bound(unique_node_ids.begin(),
                                      unique_node_ids.end(), old_node_id);
     submesh.element_conn[i] = static_cast<I>(it - unique_node_ids.begin());
  }
//
//      // Get the x, y, z coordinates for the nodes.
//      submesh.nodes_x.resize(unique_node_ids.size());
//      submesh.nodes_y.resize(unique_node_ids.size());
//      submesh.nodes_z.resize(unique_node_ids.size());
//      for (length_t i = 0; i < unique_node_ids.size(); ++i) {
//          length_t const node_id = static_cast<length_t>(unique_node_ids[i]);
//          submesh.nodes_x[i] = this->nodes_x[node_id];
//          submesh.nodes_y[i] = this->nodes_y[node_id];
//          submesh.nodes_z[i] = this->nodes_z[node_id];
//      }
//
//
  // size_t const num_elsets = elset_names.size();
//      // If the intersection of this elset and another elset is non-empty, then we need
//      to
//      // add the itersection as an elset and remap the elset IDs using the element_ids
//      vector.
//      // element_ids[i] is the old element id, and i is the new element id.
//      for (length_t i = 0; i < num_elsets; ++i) {
//          if (i == elset_index) continue;
//          length_t const elset_start = static_cast<length_t>(this->elset_offsets[i]);
//          length_t const elset_end   = static_cast<length_t>(this->elset_offsets[i +
//          1]); std::vector<I> intersection; std::set_intersection(element_ids.begin(),
//          element_ids.end(),
//                                this->elset_ids.begin() + elset_start,
//                                this->elset_ids.begin() + elset_end,
//                                std::back_inserter(intersection));
//          if (intersection.empty()) continue;
//          // We have an intersection. Add the elset.
//          submesh.elset_names.push_back(this->elset_names[i]);
//          if (submesh.elset_offsets.empty()) {
//              submesh.elset_offsets.push_back(0);
//          }
//          submesh.elset_offsets.push_back(submesh.elset_offsets.back() +
//          intersection.size()); for (length_t j = 0; j < intersection.size(); ++j) {
//              I const old_element_id = intersection[j];
//              auto const it = std::lower_bound(element_ids.begin(),
//                                               element_ids.end(), old_element_id);
//              submesh.elset_ids.push_back(static_cast<I>(it - element_ids.begin()));
//          }
//      }
}

// template <std::floating_point T, std::signed_integral I>
// constexpr MeshType MeshFile<T, I>::get_mesh_type() const
//{
//     MeshType mesh_type = MeshType::ERROR;
//     int identifier = 0;
//     if (this->format == MeshFileFormat::ABAQUS) {
//         auto is_tri = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(AbaqusCellType::CPS3);
//         };
//         auto is_quad = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(AbaqusCellType::CPS4);
//         };
//         auto is_tri6 = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(AbaqusCellType::CPS6);
//         };
//         auto is_quad8 = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(AbaqusCellType::CPS8);
//         };
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_tri)) {
//             identifier += 3;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_quad)) {
//             identifier += 4;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_tri6)) {
//             identifier += 6;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_quad8)) {
//             identifier += 8;
//         }
//     } else if (this->format == MeshFileFormat::XDMF) {
//         auto is_tri = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(XDMFCellType::TRIANGLE);
//         };
//         auto is_quad = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(XDMFCellType::QUAD);
//         };
//         auto is_tri6 = [](int8_t const element_type) {
//             return element_type ==
//             static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
//         };
//         auto is_quad8 = [](int8_t const element_type) {
//             return element_type == static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
//         };
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_tri)) {
//             identifier += 3;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_quad)) {
//             identifier += 4;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_tri6)) {
//             identifier += 6;
//         }
//         if (std::any_of(this->element_types.cbegin(),
//                         this->element_types.cend(),
//                         is_quad8)) {
//             identifier += 8;
//         }
//     } else {
//         Log::error("Unknown mesh format");
//     }
//     switch (identifier) {
//         case 3:
//             mesh_type = MeshType::TRI;
//             break;
//         case 4:
//             mesh_type = MeshType::QUAD;
//             break;
//         case 7:
//             mesh_type = MeshType::TRI_QUAD;
//             break;
//         case 6:
//             mesh_type = MeshType::QUADRATIC_TRI;
//             break;
//         case 8:
//             mesh_type = MeshType::QUADRATIC_QUAD;
//             break;
//         case 14:
//             mesh_type = MeshType::QUADRATIC_TRI_QUAD;
//             break;
//         default:
//             Log::error("Unknown mesh type");
//     }
//
//     return mesh_type;
// }
//
template <std::floating_point T, std::signed_integral I>
constexpr void
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

// template <std::floating_point T, std::signed_integral I>
// constexpr void MeshFile<T, I>::get_material_ids(std::vector<MaterialID> & material_ids,
//                                                 std::vector<std::string> const &
//                                                 material_names) const
//{
//     length_t const nelems = this->element_types.size();
//     material_ids.resize(nelems);
//     for (length_t i = 0; i < nelems; ++i) {
//         material_ids[i] = static_cast<MaterialID>(-1);
//     }
//
//     length_t const nmats = material_names.size();
//     for (length_t i = 0; i < nmats; ++i) {
//         std::string const & mat_name = material_names[i];
//         for (length_t j = 0; j < this->elset_names.size(); ++j) {
//             if (this->elset_names[j] == mat_name) {
//                 length_t const start = static_cast<length_t>(this->elset_offsets[j ]);
//                 length_t const end   = static_cast<length_t>(this->elset_offsets[j +
//                 1]); for (length_t k = start; k < end; ++k) {
//                     length_t const elem = static_cast<length_t>(this->elset_ids[k]);
//                     if (material_ids[elem] != -1) {
//                         Log::error("Element " + std::to_string(elem) + " has multiple
//                         materials");
//                     }
//                     material_ids[elem] = static_cast<MaterialID>(i);
//                 } // for k
//                 break;
//             } // if elset_names[j] == mat_name
//         } // for j
//     } // for i
//     if (std::any_of(material_ids.cbegin(), material_ids.cend(),
//                     [](MaterialID const mat_id) { return mat_id == -1; })) {
//         Log::warn("Some elements have no material");
//     }
// }
//
// template <std::floating_point T, std::signed_integral I>
// constexpr void MeshFile<T, I>::get_material_ids(std::vector<MaterialID> & material_ids)
// const
//{
//     std::vector<std::string> material_names;
//     this->get_material_names(material_names);
//     length_t const nmats = material_names.size();
//     if (nmats == 0) {
//         Log::error("No materials found in mesh file");
//     }
//     if (nmats > std::numeric_limits<MaterialID>::max()) {
//         Log::error("Number of materials exceeds MaterialID capacity");
//     }
//     this->get_material_ids(material_ids, material_names);
// }

} // namespace um2
