#include <um2/gmsh/model.hpp>

#if UM2_ENABLE_GMSH

namespace um2::gmsh::model
{

// For a model with physical groups of the form "Material X, Material Y, ...",
// color the entities in each material physical group according to corresponding
// Material in <materials>.
static void
colorMaterialPhysicalGroupEntities(std::vector<Material> const & materials)
{
  size_t const num_materials = materials.size();
  std::vector<std::string> material_names(num_materials);
  for (size_t i = 0; i < num_materials; ++i) {
    material_names[i] = "Material " + std::string(materials[i].name.data());
  }
  std::vector<int> ptags(num_materials, -1);
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags);
  // We only care about the highest dimension physical groups.
  int const dim_max = dimtags.back().first;
  for (auto const & dimtag : dimtags) {
    int const dim = dimtag.first;
    if (dim != dim_max) {
      continue;
    }
    int const tag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(dim, tag, name);
    if (name.starts_with("Material ")) {
      auto const it = std::find(material_names.begin(), material_names.end(), name);
      if (it != material_names.end() && *it == name) {
        ptrdiff_t const i = it - material_names.begin();
        ptags[static_cast<size_t>(i)] = tag;
      }
    }
  }
  // Color in reverse order so that the highest priority materials
  // overwrite the lower priority materials when they overlap.
  for (size_t i = num_materials; i > 0; --i) {
    int const tag = ptags[i - 1];
    if (tag != -1) {
      std::vector<int> tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim_max, tag, tags);
      gmsh::vectorpair mat_dimtags(tags.size());
      for (size_t j = 0; j < tags.size(); ++j) {
        mat_dimtags[j] = {dim_max, tags[j]};
      }
      Color const color = materials[i - 1].color;
      gmsh::model::setColor(mat_dimtags, static_cast<int>(color.r()),
                            static_cast<int>(color.g()), static_cast<int>(color.b()),
                            static_cast<int>(color.a()), /*recursive=*/true);
    }
  }
}
// -----------------------------------------------------------------------------

void
addToPhysicalGroup(int const dim, std::vector<int> const & tags, int const tag,
                   std::string const & name)
{
  Log::debug("Adding entities to physical group \"" + name + "\"");
  assert(std::is_sorted(tags.begin(), tags.end()));
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags, dim);
  for (auto const & existing_group_dimtag : dimtags) {
    int const existing_group_tag = existing_group_dimtag.second;
    std::string existing_group_name;
    gmsh::model::getPhysicalName(dim, existing_group_tag, existing_group_name);
    if (existing_group_name == name) {
      std::vector<int> existing_tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim, existing_group_tag, existing_tags);
      std::vector<int> new_tags(tags.size() + existing_tags.size());
      std::merge(tags.begin(), tags.end(), existing_tags.begin(), existing_tags.end(),
                 new_tags.begin());
      gmsh::model::removePhysicalGroups({
          {dim, existing_group_tag}
      });
      int const new_tag =
          gmsh::model::addPhysicalGroup(dim, new_tags, existing_group_tag, name);
      assert(new_tag == existing_group_tag);
      return;
    }
  }
  // If we get here, the physical group does not exist yet.
  gmsh::model::addPhysicalGroup(dim, tags, tag, name);
}

void
getMaterials(std::vector<Material> & materials)
{
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags);
  for (auto const & dimtag : dimtags) {
    int const dim = dimtag.first;
    int const tag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(dim, tag, name);
    if (name.starts_with("Material")) {
      std::vector<int> tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags);
      std::string const material_name = name.substr(9);
      auto const it = std::find_if(materials.begin(), materials.end(),
                                   [&material_name](Material const & material) {
                                     return material.name == material_name;
                                   });
      if (it == materials.end()) {
        // Get the color of the first entity in the physical group.
        int r = 0;
        int g = 0;
        int b = 0;
        int a = 0;
        gmsh::model::getColor(dim, tags[0], r, g, b, a);
        materials.emplace_back(ShortString(material_name.c_str()), Color(r, g, b, a));
      }
    }
  }
}

namespace occ
{

static auto
groupPreservingInputChecking(gmsh::vectorpair const & object_dimtags,
                             gmsh::vectorpair const & tool_dimtags) -> int
{
  // Ensure that the dimtags are non-empty and sorted.
  if (object_dimtags.empty() || tool_dimtags.empty()) {
    Log::error("object_dimtags or tool_dimtags is empty");
  }
  if (!std::is_sorted(object_dimtags.begin(), object_dimtags.end())) {
    Log::error("object_dimtags is not sorted");
  }
  if (!std::is_sorted(tool_dimtags.begin(), tool_dimtags.end())) {
    Log::error("tool_dimtags is not sorted");
  }

  // Ensure that the dimtags are unique. Since they are sorted, we can just
  // check that element i != element i + 1.
  for (size_t i = 0; i < object_dimtags.size() - 1; ++i) {
    if (object_dimtags[i] == object_dimtags[i + 1]) {
      Log::error("object_dimtags are not unique");
    }
  }
  for (size_t i = 0; i < tool_dimtags.size() - 1; ++i) {
    if (tool_dimtags[i] == tool_dimtags[i + 1]) {
      Log::error("tool_dimtags are not unique");
    }
  }

  // Ensure that object_dimtags contains only 2D xor 3D entities.
  int const object_dim_front = object_dimtags.front().first;
  int const object_dim_back = object_dimtags.back().first;
  if (object_dim_front != object_dim_back) {
    Log::error("object_dimtags contains entities of mixed dimension");
  }
  int const object_dim = object_dim_front;

  // Ensure that tool_dimtags contains only 2D xor 3D entities.
  int const tool_dim_front = tool_dimtags.front().first;
  int const tool_dim_back = tool_dimtags.back().first;
  if (tool_dim_front != tool_dim_back) {
    Log::error("tool_dimtags contains entities of mixed dimension");
  }
  int const tool_dim = tool_dim_front;

  if (object_dim != tool_dim) {
    Log::error("object_dimtags and tool_dimtags are not the same dimension");
  }
  if (object_dim != 2 && object_dim != 3) {
    Log::error("object_dimtags and tool_dimtags are not 2D or 3D");
  }
  return object_dim;
}

static void
getPhysicalGroupInfo(std::vector<std::string> & physical_group_names,
                     std::vector<int> & physical_group_tag,
                     std::vector<std::vector<int>> & physical_group_ent_tags,
                     gmsh::vectorpair & physical_group_dimtags, int const model_dim)
{
  gmsh::model::getPhysicalGroups(physical_group_dimtags);
  for (auto const & dimtag : physical_group_dimtags) {
    int const dim = dimtag.first;
    if (dim != model_dim) {
      continue;
    }
    int const ptag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(dim, ptag, name);
    std::vector<int> tags;
    gmsh::model::getEntitiesForPhysicalGroup(dim, ptag, tags);
    // Insert the physical group, keeping the vectors sorted
    auto const it = std::lower_bound(physical_group_names.cbegin(),
                                     physical_group_names.cend(), name);
    ptrdiff_t const idx = it - physical_group_names.cbegin();
    physical_group_names.insert(it, name);
    physical_group_tag.insert(physical_group_tag.begin() + idx, ptag);
    physical_group_ent_tags.insert(physical_group_ent_tags.begin() + idx, tags);
  }
}

static void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
getNewPhysicalGroups(gmsh::vectorpair const & object_dimtags,
                     gmsh::vectorpair const & tool_dimtags, size_t const num_groups,
                     std::vector<std::vector<int>> const & pre_physical_group_ent_tags,
                     std::vector<gmsh::vectorpair> const & out_dimtags_map,
                     std::vector<std::vector<int>> & post_physical_group_ent_tags)
{
  // For each physical group,
  //   For each entity in the group,
  //     If the entity is in the object or tool,
  //       Add the fragment children of the entity to the new physical group.
  //     Else,
  //       Add the entity to the new physical group.
  std::vector<int> object_tags(object_dimtags.size());
  std::vector<int> tool_tags(tool_dimtags.size());
  size_t const nobject = object_dimtags.size();
  size_t const ntool = tool_dimtags.size();
  for (size_t i = 0; i < object_dimtags.size(); ++i) {
    object_tags[i] = object_dimtags[i].second;
  }
  for (size_t i = 0; i < tool_dimtags.size(); ++i) {
    tool_tags[i] = tool_dimtags[i].second;
  }
  for (size_t i = 0; i < num_groups; ++i) {
    std::vector<int> const & tags = pre_physical_group_ent_tags[i];
    std::vector<int> new_tags;
    new_tags.reserve(tags.size());
    // The object, tool, and physical group entities are sorted.
    // Therefore, we can simply iterate through the entities in the
    // physical group and the object and tool entities, and compare
    // the tags.
    size_t object_idx = 0;
    size_t tool_idx = 0;
    size_t idx = 0;
    for (int const etag : tags) {
      bool found = false;
      while (object_idx < nobject && object_tags[object_idx] < etag) {
        ++object_idx;
      }
      if (object_idx < nobject && object_tags[object_idx] == etag) {
        idx = object_idx;
        found = true;
      }
      if (!found) {
        while (tool_idx < ntool && tool_tags[tool_idx] < etag) {
          ++tool_idx;
        }
        if (tool_idx < ntool && tool_tags[tool_idx] == etag) {
          idx = tool_idx + nobject;
          found = true;
        }
      }
      if (found) {
        // The entity is in the object or tool, so add the fragment children
        // to the new physical group.
        for (auto const & child : out_dimtags_map[idx]) {
          auto const child_it =
              std::lower_bound(new_tags.begin(), new_tags.end(), child.second);
          if (child_it == new_tags.end() || *child_it != child.second) {
            new_tags.insert(child_it, child.second);
          }
        }
      } else {
        // The entity is not in the object or tool, so add the entity to the
        // new physical group.
        auto const child_it = std::lower_bound(new_tags.begin(), new_tags.end(), etag);
        if (child_it == new_tags.end() || *child_it != etag) {
          new_tags.insert(child_it, etag);
        }
      }
    } // end for (int const etag : tags)
    // Insert the physical group, keeping the vector sorted by name
    post_physical_group_ent_tags[i] = new_tags;
  } // end for (size_t i = 0; i < physical_group_names.size(); ++i)
}

static void
processMaterialHierarchy(std::vector<Material> const & material_hierarchy,
                         std::vector<std::string> const & physical_group_names,
                         std::vector<std::vector<int>> & post_physical_group_ent_tags)
{
  // Process the material hierarchy, if it exists, so that each entity has one or
  // fewer materials.
  if (!material_hierarchy.empty()) {
    Log::info("Processing material hierarchy");
    // Let i in [1, nmats]
    // For each material i,
    //     If {material i} ∩ {material j} ≠ ∅ ,
    //          for j < i, {material i} = {material i} \ {material j}
    size_t const nmats = material_hierarchy.size();
    constexpr size_t guard = std::numeric_limits<size_t>::max();
    std::vector<size_t> mat_indices(nmats, guard);
    for (size_t i = 0; i < nmats; ++i) {
      Material const & mat = material_hierarchy[i];
      std::string const & mat_name = "Material " + std::string(mat.name.data());
      auto const it = std::lower_bound(physical_group_names.begin(),
                                       physical_group_names.end(), mat_name);
      if (it == physical_group_names.end() || *it != mat_name) {
        Log::warn("'Material " + std::string(mat.name.data()) + "' not found in model");
      } else {
        mat_indices[i] = static_cast<size_t>(it - physical_group_names.begin());
      }
    }
    for (size_t i = 1; i < nmats; ++i) {
      if (mat_indices[i] == guard) {
        continue;
      }
      std::vector<int> & mat_i = post_physical_group_ent_tags[mat_indices[i]];
      for (size_t j = 0; j < i; ++j) {
        if (mat_indices[j] == guard) {
          continue;
        }
        std::vector<int> & mat_j = post_physical_group_ent_tags[mat_indices[j]];
        std::vector<int> intersection;
        // We anticipate that the intersection will be empty, so we don't
        // start with a set_difference.
        std::set_intersection(mat_i.begin(), mat_i.end(), mat_j.begin(), mat_j.end(),
                              std::back_inserter(intersection));
        if (!intersection.empty()) {
          std::vector<int> difference;
          std::set_difference(mat_i.begin(), mat_i.end(), intersection.begin(),
                              intersection.end(), std::back_inserter(difference));
          mat_i = difference;
        }
      }
    }
  } // material hierarchy
}

// A gmsh::model::occ::fragment that preserves the model's D-dimensional physical
// groups when fragmenting D-dimensional entities. All other physical groups are
// destroyed.
//
// In the event that two overlapping entities have material physical groups, the
// optional material hierarchy is used to choose a single material for the
// resultant overlapping entity/entities.
void
groupPreservingFragment(gmsh::vectorpair const & object_dimtags,
                        gmsh::vectorpair const & tool_dimtags,
                        gmsh::vectorpair & out_dimtags,
                        std::vector<gmsh::vectorpair> & out_dimtags_map,
                        std::vector<Material> const & material_hierarchy, int const tag,
                        bool const remove_object, bool const remove_tool)
{

  // ----------------------------------------------------------------------
  // Input checking
  // ----------------------------------------------------------------------
  int const model_dim = groupPreservingInputChecking(object_dimtags, tool_dimtags);

  // ----------------------------------------------------------------------
  // Fragment the object and tool
  // ----------------------------------------------------------------------
  // Get the physical groups of the object and tool to preserve them.
  // They are destroyed by the fragment operation.
  std::vector<std::string> physical_group_names;
  std::vector<int> physical_group_tag;
  std::vector<std::vector<int>> pre_physical_group_ent_tags;
  gmsh::vectorpair pre_dimtags;
  getPhysicalGroupInfo(physical_group_names, physical_group_tag,
                       pre_physical_group_ent_tags, pre_dimtags, model_dim);
  size_t const num_groups = physical_group_names.size();
  size_t const nobject = object_dimtags.size();
  size_t const ntool = tool_dimtags.size();

  Log::info("Fragmenting " + std::to_string(nobject) + " object entities and " +
            std::to_string(ntool) + " tool entities");
  gmsh::model::removePhysicalGroups();

  gmsh::model::occ::fragment(object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map,
                             tag, remove_object, remove_tool);
  // Remove object or tool entities if requested.
  if (remove_object) {
    gmsh::model::removeEntities(object_dimtags, /*recursive=*/true);
  }
  if (remove_tool) {
    gmsh::model::removeEntities(tool_dimtags, /*recursive=*/true);
  }
  gmsh::model::occ::synchronize();

  // ----------------------------------------------------------------------
  // Create the new physical groups
  // ----------------------------------------------------------------------
  Log::info("Processing physical group changes");
  std::vector<std::vector<int>> post_physical_group_ent_tags(num_groups);
  getNewPhysicalGroups(object_dimtags, tool_dimtags, num_groups,
                       pre_physical_group_ent_tags, out_dimtags_map,
                       post_physical_group_ent_tags);

  // ----------------------------------------------------------------------
  // Process the material hierarchy
  // ----------------------------------------------------------------------
  processMaterialHierarchy(material_hierarchy, physical_group_names,
                           post_physical_group_ent_tags);

  // ----------------------------------------------------------------------
  // Create the new physical groups
  // ----------------------------------------------------------------------
  for (size_t i = 0; i < physical_group_names.size(); ++i) {
    int const pgroup_tag = gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
    assert(pgroup_tag == physical_group_tag[i]);
  }

  // Apply material colors.
  if (!material_hierarchy.empty()) {
    Log::info("Applying material colors");
    colorMaterialPhysicalGroupEntities(material_hierarchy);
  }
} // groupPreservingFragment

// A gmsh::model::occ::intersect that preserves the model's D-dimensional physical
// groups when fragmenting D-dimensional entities. All other physical groups are
// destroyed.
//
// In the event that two overlapping entities have material physical groups, the
// optional material hierarchy is used to choose a single material for the
// resultant overlapping entity/entities.
void
groupPreservingIntersect(gmsh::vectorpair const & object_dimtags,
                         gmsh::vectorpair const & tool_dimtags,
                         gmsh::vectorpair & out_dimtags,
                         std::vector<gmsh::vectorpair> & out_dimtags_map,
                         std::vector<Material> const & material_hierarchy, int const tag,
                         bool const remove_object, bool const remove_tool)
{

  // --------------------------------------------------------------------------
  // Input checking
  // --------------------------------------------------------------------------
  int const model_dim = groupPreservingInputChecking(object_dimtags, tool_dimtags);

  // ----------------------------------------------------------------------
  // Intersect the object and tool
  // ----------------------------------------------------------------------
  // Get the physical groups of the object and tool to preserve them.
  // They are destroyed by the fragment operation.
  std::vector<std::string> physical_group_names;
  std::vector<int> physical_group_tag;
  std::vector<std::vector<int>> pre_physical_group_ent_tags;
  gmsh::vectorpair pre_dimtags;
  getPhysicalGroupInfo(physical_group_names, physical_group_tag,
                       pre_physical_group_ent_tags, pre_dimtags, model_dim);
  size_t const num_groups = physical_group_names.size();
  size_t const nobject = object_dimtags.size();
  size_t const ntool = tool_dimtags.size();
  Log::info("Intersecting " + std::to_string(nobject) + " object entities and " +
            std::to_string(ntool) + " tool entities");
  gmsh::model::removePhysicalGroups();
  gmsh::model::occ::intersect(object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map,
                              tag, remove_object, remove_tool);
  // Remove object or tool entities if requested.
  if (remove_object) {
    gmsh::model::removeEntities(object_dimtags, /*recursive=*/true);
  }
  if (remove_tool) {
    gmsh::model::removeEntities(tool_dimtags, /*recursive=*/true);
  }
  gmsh::model::occ::synchronize();

  // ----------------------------------------------------------------------
  // Create the new physical groups
  // ----------------------------------------------------------------------
  Log::info("Processing physical group changes");
  std::vector<std::vector<int>> post_physical_group_ent_tags(num_groups);
  getNewPhysicalGroups(object_dimtags, tool_dimtags, num_groups,
                       pre_physical_group_ent_tags, out_dimtags_map,
                       post_physical_group_ent_tags);

  // ----------------------------------------------------------------------
  // Process the material hierarchy
  // ----------------------------------------------------------------------
  processMaterialHierarchy(material_hierarchy, physical_group_names,
                           post_physical_group_ent_tags);

  // ----------------------------------------------------------------------
  // Create the new physical groups
  // ----------------------------------------------------------------------
  for (size_t i = 0; i < physical_group_names.size(); ++i) {
    int const pgroup_tag = gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
    assert(pgroup_tag == physical_group_tag[i]);
  }

  // Apply material colors.
  if (!material_hierarchy.empty()) {
    Log::info("Applying material colors");
    colorMaterialPhysicalGroupEntities(material_hierarchy);
  }
} // groupPreservingIntersect

auto
addCylindricalPin2D(Point2d const & center, std::vector<double> const & radii,
                    std::vector<Material> const & materials) -> std::vector<int>
{
  Log::info("Adding 2D cylindrical pin");
  // Input checking
  size_t const nradii = radii.size();
  if (nradii == 0) {
    Log::error("radii must not be empty");
  }
  if (nradii > materials.size()) {
    Log::error("Number of radii must be <= number of materials");
  }
  if (radii[0] <= 0.0) {
    Log::error("radii must be positive");
  }
  for (size_t i = 1; i < nradii; ++i) {
    if (radii[i] <= radii[i - 1]) {
      Log::error("radii must be strictly increasing");
    }
  }
  // Create the pin geometry
  std::vector<int> out_tags;
  out_tags.reserve(nradii);
  double const x = center[0];
  double const y = center[1];
  // Do the innermost disk
  int const circle0_tag = gmsh::model::occ::addCircle(x, y, 0.0, radii[0]);
  int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
  int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
  out_tags.emplace_back(disk_tag);
  // Do the annuli
  int prev_loop_tag = loop0_tag;
  for (size_t i = 1; i < nradii; ++i) {
    int const circle_tag = gmsh::model::occ::addCircle(x, y, 0.0, radii[i]);
    int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
    int const annulus_tag = gmsh::model::occ::addPlaneSurface({loop_tag, prev_loop_tag});
    out_tags.emplace_back(annulus_tag);
    prev_loop_tag = loop_tag;
  }
  gmsh::model::occ::synchronize();
  // Add materials
  for (size_t i = 0; i < nradii; ++i) {
    addToPhysicalGroup(2, {out_tags[i]}, -1,
                       "Material " + std::string(materials[i].name.data()));
    // Color entities according to materials
    Color const color = materials[i].color;
    gmsh::model::setColor(
        {
            {2, out_tags[i]}
    }, // Entities to color
        static_cast<int>(color.r()), static_cast<int>(color.g()),
        static_cast<int>(color.b()), static_cast<int>(color.a()),
        /*recursive=*/true);
  }
  return out_tags;
}
//
//     std::vector<int> add_2d_cylindrical_pin_lattice(
//             std::vector<std::vector<double>> const & radii,
//             std::vector<std::vector<Material>> const & materials,
//             std::vector<Vec2d> const & dxdy,
//             std::vector<std::vector<int>> const & pin_ids,
//             Point2d const & offset)
//     {
//         // Possibility for size_t to 32-bit int conversion warning.
//         // This shouldn't be a problem, since the number of pins should be small.
//         Log::info("Adding 2D cylindrical pin lattice");
//         std::vector<int> out_tags;
//         // Input checking
//         size_t nunique_pins = radii.size();
//         if (nunique_pins == 0) { Log::error("radii must not be empty"); }
//         if (nunique_pins != materials.size()) {
//             Log::error("Number of radii vectors must be equal to number of material
//             vectors");
//         }
//         if (nunique_pins != dxdy.size()) {
//             Log::error("Number of radii vectors must be equal to number of dxdy
//             pairs");
//         }
//         // Ensure each row of pin_ids has the same length
//         size_t ncol = pin_ids[0].size();
//         for (size_t i = 1; i < pin_ids.size(); ++i) {
//             if (pin_ids[i].size() != ncol) {
//                 Log::error("Each row of pin_ids must have the same length");
//             }
//         }
//         // Ensure all pin_ids are in range 0:nunique_pins - 1.
//         for (size_t i = 0; i < pin_ids.size(); ++i) {
//             for (size_t j = 0; j < pin_ids[i].size(); ++j) {
//                 if (pin_ids[i][j] < 0 || static_cast<size_t>(pin_ids[i][j]) >=
//                 nunique_pins) {
//                     Log::error("pin_ids must be in range [0, nunique_pins - 1]");
//                 }
//             }
//         }
//         size_t nrow = pin_ids.size();
//         size_t npins = nrow * ncol;
//         // Reverse the ordering of the rows of pin_ids, so the vector ordering
//         matches the
//         // spatial ordering.
//         std::vector<std::vector<int>> pin_ids_rev(nrow);
//         for (size_t i = 0; i < nrow; ++i) {
//             pin_ids_rev[i] = pin_ids[nrow - i - 1];
//         }
//         // Construct a RectilinearGrid object using the vector of AABox constructor.
//         std::vector<AABox2d> boxes(npins);
//         double ymin = 0.0;
//         for (size_t i = 0; i < nrow; ++i) {
//             double xmin = 0.0;
//             std::vector<int> const & row = pin_ids_rev[i];
//             int pin_id = row[0];
//             unsigned pin_idx = static_cast<unsigned>(pin_id);
//             double const ymax = ymin + dxdy[pin_idx][1u];
//             for (size_t j = 0; j < ncol; ++j) {
//                 pin_id = row[j];
//                 pin_idx = static_cast<unsigned>(pin_id);
//                 double const xmax = xmin + dxdy[pin_idx][0u];
//                 boxes[i * ncol + j] = AABox2d(Point2d(xmin, ymin), Point2d(xmax,
//                 ymax)); xmin = xmax;
//             }
//             ymin = ymax;
//         }
//
//         RectilinearGrid2d grid(boxes.data(), boxes.size());
//
//         // For each unique pin, loop through the pin_ids_rev array and add the pins.
//         std::vector<std::vector<int>> pin_tags(nunique_pins);
//         for (size_t pin_id = 0; pin_id < nunique_pins; ++pin_id) {
//             size_t const nrad = radii[pin_id].size();
//             std::vector<std::vector<int>> material_ids(nrad);
//             for (size_t i = 0; i < nrow; ++i) {
//                 std::vector<int> const & row = pin_ids_rev[i];
//                 for (size_t j = 0; j < ncol; ++j) {
//                     if (row[j] == static_cast<int>(pin_id)) {
//                         AABox2d box = grid.get_box(i, j);
//                         double const x = 0.5 * (x_min(box) + x_max(box)) +
//                         offset[0u]; double const y = 0.5 * (y_min(box) + y_max(box))
//                         + offset[1u];
//                         // Do the innermost disk
//                         double const r0 = radii[pin_id][0];
//                         int const circle0_tag = gmsh::model::occ::addCircle(x, y,
//                         0.0, r0); int const loop0_tag =
//                         gmsh::model::occ::addCurveLoop({circle0_tag}); int const
//                         disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
//                         material_ids[0].emplace_back(disk_tag);
//                         out_tags.emplace_back(disk_tag);
//                         // Do the annuli
//                         int prev_loop_tag = loop0_tag;
//                         for (size_t k = 1; k < nrad; ++k) {
//                             double const r = radii[pin_id][k];
//                             int const circle_tag = gmsh::model::occ::addCircle(x, y,
//                             0.0, r); int const loop_tag =
//                             gmsh::model::occ::addCurveLoop({circle_tag}); int const
//                             annulus_tag =
//                                 gmsh::model::occ::addPlaneSurface({loop_tag,
//                                 prev_loop_tag});
//                             material_ids[k].emplace_back(annulus_tag);
//                             out_tags.emplace_back(annulus_tag);
//                             prev_loop_tag = loop_tag;
//                         }
//                     }
//                 }
//             }
//             gmsh::model::occ::synchronize();
//             // Add materials
//             for (size_t i = 0; i < nrad; ++i) {
//                 size_t const nents = material_ids[i].size();
//                 if (nents == 0) { continue; }
//                 add_to_physical_group(
//                         2, // dim
//                         material_ids[i], // tags
//                         -1, // tag
//                         "Material " + to_string(materials[pin_id][i].name) // name
//                         );
//                 // Color entities according to materials
//                 gmsh::vectorpair mat_dimtags(nents);
//                 for (size_t j = 0; j < nents; ++j) {
//                     mat_dimtags[j] = std::make_pair(2, material_ids[i][j]);
//                 }
//                 Color const color = materials[pin_id][i].color;
//                 gmsh::model::setColor(
//                         mat_dimtags,
//                         static_cast<int>(color.r),
//                         static_cast<int>(color.g),
//                         static_cast<int>(color.b),
//                         static_cast<int>(color.a),
//                         true);
//             }
//         } // end loop over unique pins
//         return out_tags;
//
//     } // end function
//
//     std::vector<int> add_cylindrical_pin(
//             Point3d const & center,
//             double const height,
//             std::vector<double> const & radii,
//             std::vector<Material> const & materials)
//     {
//         Log::info("Adding cylindrical pin");
//         // Input checking
//         size_t nradii = radii.size();
//         if (height <= 0.0) { Log::error("height must be positive"); }
//         if (nradii == 0) { Log::error("radii must not be empty"); }
//         if (nradii > materials.size()) {
//             Log::error("Number of radii must be <= number of materials");
//         }
//         if (radii[0] <= 0.0) { Log::error("radii must be positive"); }
//         for (size_t i = 1; i < nradii; ++i) {
//             if (radii[i] <= radii[i-1]) { Log::error("radii must be strictly
//             increasing"); }
//         }
//         // Create the pin geometry
//         gmsh::vectorpair dimtags_2d;
//         double const x = center[0u];
//         double const y = center[1u];
//         double const z = center[2u];
//         // Do the innermost disk
//         int const circle0_tag = gmsh::model::occ::addCircle(x, y, z, radii[0]);
//         int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
//         int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
//         dimtags_2d.push_back({2, disk_tag});
//         // Do the annuli
//         int prev_loop_tag = loop0_tag;
//         for (size_t i = 1; i < nradii; ++i) {
//             int const circle_tag = gmsh::model::occ::addCircle(x, y, z, radii[i]);
//             int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
//             int const annulus_tag = gmsh::model::occ::addPlaneSurface({loop_tag,
//             prev_loop_tag}); dimtags_2d.push_back({2, annulus_tag}); prev_loop_tag =
//             loop_tag;
//         }
//         gmsh::vectorpair out_dimtags;
//         gmsh::model::occ::extrude(dimtags_2d, 0.0, 0.0, height, out_dimtags);
//         gmsh::model::occ::synchronize();
//         std::vector<int> out_tags;
//         out_tags.reserve(nradii);
//         for (auto const & dt : out_dimtags) {
//             if (dt.first == 3) { out_tags.emplace_back(dt.second); }
//         }
//         // Add materials
//         for (size_t i = 0; i < nradii; ++i) {
//             add_to_physical_group(3, {out_tags[i]}, -1, "Material " +
//             to_string(materials[i].name));
//             // Color entities according to materials
//             Color const color = materials[i].color;
//             gmsh::model::setColor(
//                     {{3, out_tags[i]}}, // Entities to color
//                     static_cast<int>(color.r),
//                     static_cast<int>(color.g),
//                     static_cast<int>(color.b),
//                     static_cast<int>(color.a),
//                     true);
//         }
//         return out_tags;
//     } // end function
//
//     std::vector<int> add_cylindrical_pin_lattice(
//             std::vector<std::vector<double>> const & radii,
//             std::vector<std::vector<Material>> const & materials,
//             double const height,
//             std::vector<Vec2d> const & dxdy,
//             std::vector<std::vector<int>> const & pin_ids,
//             Point3d const & offset)
//     {
//         // Possibility for size_t to 32-bit int conversion warning.
//         // This shouldn't be a problem, since the number of pins should be small.
//
//         Log::info("Adding cylindrical pin lattice");
//         // Input checking
//         size_t nunique_pins = radii.size();
//         if (height <= 0.0) { Log::error("height must be positive"); }
//         if (nunique_pins == 0) { Log::error("radii must not be empty"); }
//         if (nunique_pins != materials.size()) {
//             Log::error("Number of radii vectors must be equal to number of material
//             vectors");
//         }
//         if (nunique_pins != dxdy.size()) {
//             Log::error("Number of radii vectors must be equal to number of dxdy
//             pairs");
//         }
//         // Ensure each row of pin_ids has the same length
//         size_t ncol = pin_ids[0].size();
//         for (size_t i = 1; i < pin_ids.size(); ++i) {
//             if (pin_ids[i].size() != ncol) {
//                 Log::error("Each row of pin_ids must have the same length");
//             }
//         }
//         size_t nrow = pin_ids.size();
//         size_t npins = nrow * ncol;
//         // Reverse the ordering of the rows of pin_ids, so the vector ordering
//         matches the
//         // spatial ordering.
//         std::vector<std::vector<int>> pin_ids_rev(nrow);
//         for (size_t i = 0; i < nrow; ++i) {
//             pin_ids_rev[i] = pin_ids[nrow - i - 1];
//         }
//         // Construct a RectilinearGrid object using the vector of AABox constructor.
//         std::vector<AABox2d> boxes(npins);
//         double ymin = 0.0;
//         for (size_t i = 0; i < nrow; ++i) {
//             double xmin = 0.0;
//             std::vector<int> const & row = pin_ids_rev[i];
//             int pin_id = row[0];
//             unsigned pin_idx = static_cast<unsigned>(pin_id);
//             double const ymax = ymin + dxdy[pin_idx][1u];
//             for (size_t j = 0; j < ncol; ++j) {
//                 pin_id = row[j];
//                 pin_idx = static_cast<unsigned>(pin_id);
//                 double const xmax = xmin + dxdy[pin_idx][0u];
//                 boxes[i * ncol + j] = AABox2d(Point2d(xmin, ymin), Point2d(xmax,
//                 ymax)); xmin = xmax;
//             }
//             ymin = ymax;
//         }
//         RectilinearGrid2d grid(boxes.data(), boxes.size());
//
//         std::vector<int> out_tags;
//         // For each unique pin, loop through the pin_ids_rev array and add the pins.
//         std::vector<std::vector<int>> pin_tags(nunique_pins);
//         for (size_t pin_id = 0; pin_id < nunique_pins; ++pin_id) {
//             size_t const nrad = radii[pin_id].size();
//             std::vector<std::vector<int>> material_ids(nrad);
//             for (size_t i = 0; i < nrow; ++i) {
//                 std::vector<int> const & row = pin_ids_rev[i];
//                 for (size_t j = 0; j < ncol; ++j) {
//                     if (row[j] == static_cast<int>(pin_id)) {
//                         AABox2d box = grid.get_box(i, j);
//                         double const x = 0.5 * (x_min(box) + x_max(box)) +
//                         offset[0u]; double const y = 0.5 * (y_min(box) + y_max(box))
//                         + offset[1u]; double const z = offset[2u];
//                         // Do the innermost disk
//                         double const r0 = radii[pin_id][0];
//                         int const circle0_tag = gmsh::model::occ::addCircle(x, y, z,
//                         r0); int const loop0_tag =
//                         gmsh::model::occ::addCurveLoop({circle0_tag}); int const
//                         disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
//                         material_ids[0].emplace_back(disk_tag);
//                         // Do the annuli
//                         int prev_loop_tag = loop0_tag;
//                         for (size_t k = 1; k < nrad; ++k) {
//                             double const r = radii[pin_id][k];
//                             int const circle_tag = gmsh::model::occ::addCircle(x, y,
//                             z, r); int const loop_tag =
//                             gmsh::model::occ::addCurveLoop({circle_tag}); int const
//                             annulus_tag =
//                                 gmsh::model::occ::addPlaneSurface({loop_tag,
//                                 prev_loop_tag});
//                             material_ids[k].emplace_back(annulus_tag);
//                             prev_loop_tag = loop_tag;
//                         } // nrad
//                     } // row[j] == static_cast<int>(pin_id)
//                 } // ncol
//             } // nrow
//             gmsh::model::occ::synchronize();
//             // Extrude and add materials
//             for (size_t i = 0; i < nrad; ++i) {
//                 size_t const nents = material_ids[i].size();
//                 if (nents == 0) { continue; }
//                 gmsh::vectorpair out_dimtags;
//                 gmsh::vectorpair in_dimtags(nents);
//                 for (size_t j = 0; j < nents; ++j) {
//                     in_dimtags[j] = std::make_pair(2, material_ids[i][j]);
//                 }
//                 gmsh::model::occ::extrude(in_dimtags, 0.0, 0.0, height, out_dimtags);
//                 gmsh::model::occ::synchronize();
//                 // Overwrite the material ids with the extruded tags.
//                 size_t ctr = 0;
//                 for (auto const & dimtag : out_dimtags) {
//                     if (dimtag.first == 3) {
//                         material_ids[i][ctr] = dimtag.second;
//                         out_tags.emplace_back(dimtag.second);
//                         ++ctr;
//                     }
//                 }
//                 add_to_physical_group(
//                         3, // dim
//                         material_ids[i], // tags
//                         -1, // tag
//                         "Material " + to_string(materials[pin_id][i].name) // name
//                         );
//                 // Color entities according to materials
//                 gmsh::vectorpair mat_dimtags(nents);
//                 for (size_t j = 0; j < nents; ++j) {
//                     mat_dimtags[j] = std::make_pair(3, material_ids[i][j]);
//                 }
//                 Color const color = materials[pin_id][i].color;
//                 gmsh::model::setColor(
//                         mat_dimtags,
//                         static_cast<int>(color.r),
//                         static_cast<int>(color.g),
//                         static_cast<int>(color.b),
//                         static_cast<int>(color.a),
//                         true // recursive
//                         );
//             } // nrad
//         } // end loop over unique pins
//         return out_tags;
//     } // end function
//
} // namespace occ
} // namespace um2::gmsh::model
#endif // UM2_ENABLE_GMSH
