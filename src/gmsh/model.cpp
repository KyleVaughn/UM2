#include <um2/config.hpp>

#if UM2_USE_GMSH

#  include <um2/common/cast_if_not.hpp>
#  include <um2/common/color.hpp>
#  include <um2/common/logger.hpp>
#  include <um2/geometry/axis_aligned_box.hpp>
#  include <um2/geometry/point.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/model.hpp>
#  include <um2/math/vec.hpp>
#  include <um2/mesh/rectilinear_grid.hpp>
#  include <um2/mpact/model.hpp>
#  include <um2/physics/material.hpp>
#  include <um2/stdlib/algorithm/is_sorted.hpp>
#  include <um2/stdlib/assert.hpp>
#  include <um2/stdlib/string.hpp>
#  include <um2/stdlib/utility/move.hpp>
#  include <um2/stdlib/vector.hpp>

#  include <algorithm>
#  include <cstddef>
#  include <cstdint>
#  include <iterator>
#  include <limits>
#  include <string>
#  include <utility>
#  include <vector>

namespace um2::gmsh::model
{

//=============================================================================
// addToPhysicalGroup
//=============================================================================

void
addToPhysicalGroup(int const dim, std::vector<int> const & tags, int const tag,
                   std::string const & name)
{
  LOG_DEBUG("Adding entities to physical group \"", name.c_str(), "\"");
  ASSERT(um2::is_sorted(tags.begin(), tags.end()));
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
#  if UM2_ENABLE_ASSERTS
      int const new_tag =
          gmsh::model::addPhysicalGroup(dim, new_tags, existing_group_tag, name);
      ASSERT(new_tag == existing_group_tag);
#  else
      gmsh::model::addPhysicalGroup(dim, new_tags, existing_group_tag, name);
#  endif
      return;
    }
  }
  // If we get here, the physical group does not exist yet.
  gmsh::model::addPhysicalGroup(dim, tags, tag, name);
}

//=============================================================================
// getMaterials
//=============================================================================

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
      um2::String const material_name(name.substr(9).c_str());
      auto const it = std::find_if(materials.begin(), materials.end(),
                                   [&material_name](Material const & material) {
                                     return material.getName() == material_name;
                                   });
      if (it == materials.end()) {
        // Get the color of the first entity in the physical group.
        int r = 0;
        int g = 0;
        int b = 0;
        int a = 0;
        gmsh::model::getColor(dim, tags[0], r, g, b, a);
        Material mat;
        mat.setName(material_name);
        mat.setColor(Color(r, g, b, a));
        materials.push_back(um2::move(mat));
      }
    }
  }
}

namespace occ
{

//=============================================================================
// colorMaterialPhysicalGroupEntities
//=============================================================================
//
// For a model with physical groups of the form "Material_X, Material_Y, ...",
// color the entities in each material physical group according to corresponding
// Material_in <materials>.

void
colorMaterialPhysicalGroupEntities(std::vector<Material> const & materials)
{
  size_t const num_materials = materials.size();
  std::vector<std::string> material_names(num_materials);
  for (size_t i = 0; i < num_materials; ++i) {
    material_names[i] = "Material_" + std::string(materials[i].getName().data());
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
    if (name.starts_with("Material_")) {
      auto const it = std::find(material_names.begin(), material_names.end(), name);
      if (it != material_names.end() && *it == name) {
        ptrdiff_t const i = it - material_names.begin();
        ptags[static_cast<size_t>(i)] = tag;
      }
    }
  }
  // Color in reverse order so that the highest priority materials
  // overwrite the lower priority materials. This is mainly for the
  // lower dimensional entities that are colored recursively.
  for (size_t i = num_materials; i > 0; --i) {
    int const tag = ptags[i - 1];
    if (tag != -1) {
      std::vector<int> tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim_max, tag, tags);
      gmsh::vectorpair mat_dimtags(tags.size());
      for (size_t j = 0; j < tags.size(); ++j) {
        mat_dimtags[j] = {dim_max, tags[j]};
      }
      Color const color = materials[i - 1].getColor();
      gmsh::model::setColor(mat_dimtags, static_cast<int>(color.r()),
                            static_cast<int>(color.g()), static_cast<int>(color.b()),
                            static_cast<int>(color.a()), /*recursive=*/true);
    }
  }
}

namespace
{
//=============================================================================
// groupPreservingInputChecking
//=============================================================================

auto
groupPreservingInputChecking(gmsh::vectorpair const & object_dimtags,
                             gmsh::vectorpair const & tool_dimtags) -> int
{
  // Ensure that the dimtags are non-empty and sorted.
  if (object_dimtags.empty() || tool_dimtags.empty()) {
    LOG_ERROR("object_dimtags or tool_dimtags is empty");
    return -1;
  }
  if (!std::is_sorted(object_dimtags.begin(), object_dimtags.end())) {
    LOG_ERROR("object_dimtags is not sorted");
    return -1;
  }
  if (!std::is_sorted(tool_dimtags.begin(), tool_dimtags.end())) {
    LOG_ERROR("tool_dimtags is not sorted");
    return -1;
  }

  // Ensure that the dimtags are unique. Since they are sorted, we can just
  // check that element i != element i + 1.
  for (size_t i = 0; i < object_dimtags.size() - 1; ++i) {
    if (object_dimtags[i] == object_dimtags[i + 1]) {
      LOG_ERROR("object_dimtags are not unique");
      return -1;
    }
  }
  for (size_t i = 0; i < tool_dimtags.size() - 1; ++i) {
    if (tool_dimtags[i] == tool_dimtags[i + 1]) {
      LOG_ERROR("tool_dimtags are not unique");
      return -1;
    }
  }

  // Ensure that object_dimtags contains only 2D xor 3D entities.
  int const object_dim_front = object_dimtags.front().first;
  int const object_dim_back = object_dimtags.back().first;
  if (object_dim_front != object_dim_back) {
    LOG_ERROR("object_dimtags contains entities of mixed dimension");
    return -1;
  }
  int const object_dim = object_dim_front;

  // Ensure that tool_dimtags contains only 2D xor 3D entities.
  int const tool_dim_front = tool_dimtags.front().first;
  int const tool_dim_back = tool_dimtags.back().first;
  if (tool_dim_front != tool_dim_back) {
    LOG_ERROR("tool_dimtags contains entities of mixed dimension");
    return -1;
  }
  int const tool_dim = tool_dim_front;

  if (object_dim != tool_dim) {
    LOG_ERROR("object_dimtags and tool_dimtags are not the same dimension");
    return -1;
  }
  if (object_dim != 2 && object_dim != 3) {
    LOG_ERROR("object_dimtags and tool_dimtags are not 2D or 3D");
    return -1;
  }
  return object_dim;
}

//=============================================================================
// getPhysicalGroupInfo
//=============================================================================

void
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

//=============================================================================
// getNewPhysicalGroups
//=============================================================================

void
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
          // If the child is not in the new physical group, add it.
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
  } // end for (size_t i = 0; i < num_groups; ++i)
}

//=============================================================================
// processMaterialHierarchy
//=============================================================================

void
processMaterialHierarchy(std::vector<Material> const & material_hierarchy,
                         std::vector<std::string> const & physical_group_names,
                         std::vector<std::vector<int>> & post_physical_group_ent_tags)
{
  ASSERT(um2::is_sorted(physical_group_names.begin(), physical_group_names.end()));
  // Process the material hierarchy, if it exists, so that each entity has one or
  // fewer materials.
  if (!material_hierarchy.empty()) {
    LOG_INFO("Processing material hierarchy");
    // Let i in [1, nmats]
    // For each material i,
    //     If {material i} ∩ {material j} ≠ ∅ ,
    //          for j < i, {material i} = {material i} \ {material j}
    size_t const nmats = material_hierarchy.size();
    constexpr size_t guard = std::numeric_limits<size_t>::max();
    std::vector<size_t> mat_indices(nmats, guard);
    for (size_t i = 0; i < nmats; ++i) {
      Material const & mat = material_hierarchy[i];
      std::string const & mat_name = "Material_" + std::string(mat.getName().data());
      auto const it = std::lower_bound(physical_group_names.begin(),
                                       physical_group_names.end(), mat_name);
      if (it == physical_group_names.end() || *it != mat_name) {
        LOG_WARN("'Material_", mat.getName(), "' not found in model");
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

} // namespace
//
//=============================================================================
// groupPreservingFragment
//=============================================================================
//
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

  //==============================================================================
  // Input checking
  //==============================================================================
  int const model_dim = groupPreservingInputChecking(object_dimtags, tool_dimtags);

  //==============================================================================
  // Fragment the object and tool
  //==============================================================================
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

  LOG_INFO("Fragmenting ", nobject, " object entities and ", ntool, " tool entities");
  gmsh::model::removePhysicalGroups();

  gmsh::model::occ::fragment(object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map,
                             tag, remove_object, remove_tool);
  gmsh::model::occ::synchronize();

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  LOG_INFO("Processing physical group changes");
  std::vector<std::vector<int>> post_physical_group_ent_tags(num_groups);
  getNewPhysicalGroups(object_dimtags, tool_dimtags, num_groups,
                       pre_physical_group_ent_tags, out_dimtags_map,
                       post_physical_group_ent_tags);

  //==============================================================================
  // Process the material hierarchy
  //==============================================================================
  processMaterialHierarchy(material_hierarchy, physical_group_names,
                           post_physical_group_ent_tags);

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  for (size_t i = 0; i < physical_group_names.size(); ++i) {
#  if UM2_ENABLE_ASSERTS
    int const pgroup_tag = gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
    ASSERT(pgroup_tag == physical_group_tag[i]);
#  else
    gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
#  endif
  }

  // Apply material colors.
  if (!material_hierarchy.empty()) {
    LOG_INFO("Applying material colors");
    colorMaterialPhysicalGroupEntities(material_hierarchy);
  }
} // groupPreservingFragment

//=============================================================================
// groupPreservingIntersect
//=============================================================================
//
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

  //==============================================================================-
  // Input checking
  //==============================================================================-
  int const model_dim = groupPreservingInputChecking(object_dimtags, tool_dimtags);

  //==============================================================================
  // Intersect the object and tool
  //==============================================================================
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
  LOG_INFO("Intersecting ", nobject, " object entities and ", ntool, " tool entities");
  gmsh::model::removePhysicalGroups();
  gmsh::model::occ::intersect(object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map,
                              tag, remove_object, remove_tool);
  gmsh::model::occ::synchronize();

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  LOG_INFO("Processing physical group changes");
  std::vector<std::vector<int>> post_physical_group_ent_tags(num_groups);
  getNewPhysicalGroups(object_dimtags, tool_dimtags, num_groups,
                       pre_physical_group_ent_tags, out_dimtags_map,
                       post_physical_group_ent_tags);

  //==============================================================================
  // Process the material hierarchy
  //==============================================================================
  processMaterialHierarchy(material_hierarchy, physical_group_names,
                           post_physical_group_ent_tags);

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  for (size_t i = 0; i < physical_group_names.size(); ++i) {
#  if UM2_ENABLE_ASSERTS
    int const pgroup_tag = gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
    ASSERT(pgroup_tag == physical_group_tag[i]);
#  else
    gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
#  endif
  }

  // Apply material colors.
  if (!material_hierarchy.empty()) {
    LOG_INFO("Applying material colors");
    colorMaterialPhysicalGroupEntities(material_hierarchy);
  }
} // groupPreservingIntersect

//=============================================================================
// groupPreservingCut
//=============================================================================
//
// A gmsh::model::occ::cut that preserves the model's D-dimensional physical
// groups when fragmenting D-dimensional entities. All other physical groups are
// destroyed.
//
void
groupPreservingCut(gmsh::vectorpair const & object_dimtags,
                   gmsh::vectorpair const & tool_dimtags, gmsh::vectorpair & out_dimtags,
                   std::vector<gmsh::vectorpair> & out_dimtags_map, int const tag,
                   bool const remove_object, bool const remove_tool)
{

  //==============================================================================-
  // Input checking
  //==============================================================================-
  int const model_dim = groupPreservingInputChecking(object_dimtags, tool_dimtags);

  //==============================================================================
  // Cut the object and tool
  //==============================================================================
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
  LOG_INFO("Cutting ", nobject, " object entities and ", ntool, " tool entities");
  gmsh::model::removePhysicalGroups();
  gmsh::model::occ::cut(object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map, tag,
                        remove_object, remove_tool);
  gmsh::model::occ::synchronize();

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  LOG_INFO("Processing physical group changes");
  std::vector<std::vector<int>> post_physical_group_ent_tags(num_groups);
  getNewPhysicalGroups(object_dimtags, tool_dimtags, num_groups,
                       pre_physical_group_ent_tags, out_dimtags_map,
                       post_physical_group_ent_tags);

  //==============================================================================
  // Create the new physical groups
  //==============================================================================
  for (size_t i = 0; i < physical_group_names.size(); ++i) {
#  if UM2_ENABLE_ASSERTS
    int const pgroup_tag = gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
    ASSERT(pgroup_tag == physical_group_tag[i]);
#  else
    gmsh::model::addPhysicalGroup(
        model_dim,                       // Dimension of the physical group
        post_physical_group_ent_tags[i], // Tags of the entities in the group
        physical_group_tag[i],           // Old tag of the physical group
        physical_group_names[i]);        // Name of the physical group
#  endif
  }

} // groupPreservingCut

//=============================================================================
// addCylindricalPin2D
//=============================================================================

auto
addCylindricalPin2D(Vec2F const & center, Vector<Float> const & radii,
                    Vector<Material> const & materials) -> Vector<Int>
{
  LOG_INFO("Adding 2D cylindrical pin");
  Vector<Int> out_tags;
  // Input checking
  Int const nradii = radii.size();
  if (nradii == 0) {
    LOG_ERROR("radii must not be empty");
    return out_tags;
  }
  if (nradii != materials.size()) {
    LOG_ERROR("Number of radii must equal to the number of materials");
    return out_tags;
  }
  if (radii[0] <= 0.0) {
    LOG_ERROR("radii must be positive");
    return out_tags;
  }
  if (!um2::is_sorted(radii.cbegin(), radii.cend())) {
    LOG_ERROR("radii must be strictly increasing");
    return out_tags;
  }
  // Create the pin geometry
  out_tags.reserve(nradii);
  auto const x = castIfNot<double>(center[0]);
  auto const y = castIfNot<double>(center[1]);
  // Do the innermost disk
  int const circle0_tag = gmsh::model::occ::addCircle(x, y, 0.0, radii[0]);
  int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
  int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
  out_tags.emplace_back(castIfNot<Int>(disk_tag));
  // Do the annuli
  int prev_loop_tag = loop0_tag;
  for (Int i = 1; i < nradii; ++i) {
    int const circle_tag = gmsh::model::occ::addCircle(x, y, 0.0, radii[i]);
    int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
    int const annulus_tag = gmsh::model::occ::addPlaneSurface({loop_tag, prev_loop_tag});
    out_tags.emplace_back(castIfNot<Int>(annulus_tag));
    prev_loop_tag = loop_tag;
  }
  gmsh::model::occ::synchronize();
  // Add materials
  for (Int i = 0; i < nradii; ++i) {
    ASSERT(!materials[i].getName().empty());
    addToPhysicalGroup(2, {castIfNot<int>(out_tags[i])}, -1,
                       "Material_" + std::string(materials[i].getName().data()));
    // Color entities according to materials
    Color const color = materials[i].getColor();
    gmsh::model::setColor(
        {
            {2, castIfNot<int>(out_tags[i])}
    }, // Entities to color
        static_cast<int>(color.r()), static_cast<int>(color.g()),
        static_cast<int>(color.b()), static_cast<int>(color.a()),
        /*recursive=*/true);
  }
  return out_tags;
}

// auto
// addCylindricalPin2D(Vec2d const & center, Vector<double> const & radii,
//                     Vector<Material> const & materials) -> um2::Vector<int>
//{
//   std::vector<double> radii_vec(static_cast<size_t>(radii.size()));
//   std::vector<Material> materials_vec(static_cast<size_t>(materials.size()));
//   for (size_t i = 0; i < radii_vec.size(); ++i) {
//     radii_vec[i] = radii[static_cast<Int>(i)];
//     materials_vec[i] = materials[static_cast<Int>(i)];
//   }
//   std::vector<int> out_tags = addCylindricalPin2D(center, radii_vec, materials_vec);
//   um2::Vector<int> out_tags_um2(static_cast<Int>(out_tags.size()));
//   for (Int i = 0; i < out_tags_um2.size(); ++i) {
//     out_tags_um2[i] = out_tags[static_cast<size_t>(i)];
//   }
//   return out_tags_um2;
// }
//
//=============================================================================
//  addCylindricalPinLattice2D
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
addCylindricalPinLattice2D(Vector<Vector<Int>> const & pin_ids,
                           Vector<Vec2F> const & xy_extents,
                           Vector<Vector<Float>> const & radii,
                           Vector<Vector<Material>> const & materials,
                           Vec2F const & offset) -> Vector<Int>
{
  LOG_INFO("Adding 2D cylindrical pin lattice");
  Vector<Int> out_tags;
  // Input checking
  auto const nunique_pins = radii.size();
  if (nunique_pins == 0) {
    LOG_ERROR("radii must not be empty");
    return out_tags;
  }
  if (nunique_pins != materials.size()) {
    LOG_ERROR("Number of radii vectors must be equal to number of material vectors");
    return out_tags;
  }
  if (nunique_pins != xy_extents.size()) {
    LOG_ERROR("Number of radii vectors must be equal to number of xy_extents");
    return out_tags;
  }
  // Ensure each row of pin_ids has the same length
  auto const ncol = pin_ids[0].size();
  for (auto const & row : pin_ids) {
    if (row.size() != ncol) {
      LOG_ERROR("Each row of pin_ids must have the same length");
      return out_tags;
    }
  }
  // Ensure all pin_ids are in range 0:nunique_pins - 1.
  for (auto const & row : pin_ids) {
    for (auto const & id : row) {
      if (id < 0 || id >= nunique_pins) {
        LOG_ERROR("pin_ids must be in range [0, nunique_pins - 1]");
        return out_tags;
      }
    }
  }
  auto const nrow = pin_ids.size();
  // Use the rectilinear grid to validate the correctness of the lattice and
  // get the centroids of the grid cells.
  RectilinearGrid2F const grid(xy_extents, pin_ids);

  // For each unique pin, loop through the pin_ids array and add the pins.
  for (Int pin_id = 0; pin_id < nunique_pins; ++pin_id) {
    Int const nrad = radii[pin_id].size();
    if (nrad == 0) {
      continue;
    }
    std::vector<std::vector<int>> material_ids(static_cast<size_t>(nrad));
    // Loop over rows in reverse order to go from bottom to top
    for (Int irow = nrow - 1; irow >= 0; --irow) {
      Vector<Int> const & row = pin_ids[irow];
      for (Int icol = 0; icol < ncol; ++icol) {
        if (row[icol] == pin_id) {
          auto const c = grid.getBox(icol, (nrow - 1) - irow).centroid();
          auto const x = castIfNot<double>(c[0] + offset[0]);
          auto const y = castIfNot<double>(c[1] + offset[1]);
          // Do the innermost disk
          auto const r0 = castIfNot<double>(radii[pin_id][0]);
          int const circle0_tag = gmsh::model::occ::addCircle(x, y, 0.0, r0);
          int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
          int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
          material_ids[0].emplace_back(disk_tag);
          out_tags.emplace_back(disk_tag);
          // Do the annuli
          int prev_loop_tag = loop0_tag;
          for (Int k = 1; k < nrad; ++k) {
            auto const r = castIfNot<double>(radii[pin_id][k]);
            int const circle_tag = gmsh::model::occ::addCircle(x, y, 0.0, r);
            int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
            int const annulus_tag =
                gmsh::model::occ::addPlaneSurface({loop_tag, prev_loop_tag});
            material_ids[static_cast<size_t>(k)].emplace_back(annulus_tag);
            out_tags.emplace_back(annulus_tag);
            prev_loop_tag = loop_tag;
          }
        }
      }
    }
    // Only synchronize once per unique pin
    gmsh::model::occ::synchronize();

    // Add the pin's materials
    for (size_t imat = 0; imat < static_cast<size_t>(nrad); ++imat) {
      // Number of entities for this material
      size_t const nents = material_ids[imat].size();
      auto const & material = materials[pin_id][static_cast<Int>(imat)];
      if (nents == 0) {
        continue;
      }
      addToPhysicalGroup(2,                                                   // dim
                         material_ids[imat],                                  // tags
                         -1,                                                  // tag
                         "Material_" + std::string(material.getName().data()) // name
      );
      // Color entities according to materials
      gmsh::vectorpair mat_dimtags(nents);
      for (size_t ient = 0; ient < nents; ++ient) {
        mat_dimtags[ient] = std::make_pair(2, material_ids[imat][ient]);
      }
      Color const color = material.getColor();
      gmsh::model::setColor(mat_dimtags, static_cast<int>(color.r()),
                            static_cast<int>(color.g()), static_cast<int>(color.b()),
                            static_cast<int>(color.a()), /*recursive=*/true);
    }
  } // end loop over unique pins
  return out_tags;

} // addCylindricalPinLattice2D

////==============================================================================
//// addCylindricalPin
////==============================================================================
//
// auto
// addCylindricalPin(Vec3d const & center, double const height,
//                  std::vector<double> const & radii,
//                  std::vector<Material> const & materials) -> std::vector<int>
//{
//  LOG_INFO("Adding cylindrical pin");
//  // Input checking
//  size_t const nradii = radii.size();
//  if (height <= 0.0) {
//    LOG_ERROR("height must be positive");
//  }
//  if (nradii == 0) {
//    LOG_ERROR("radii must not be empty");
//  }
//  if (nradii > materials.size()) {
//    LOG_ERROR("Number of radii must be <= number of materials");
//  }
//  if (radii[0] <= 0.0) {
//    LOG_ERROR("radii must be positive");
//  }
//  if (!std::is_sorted(radii.cbegin(), radii.cend())) {
//    LOG_ERROR("radii must be strictly increasing");
//  }
//  // Create the pin geometry
//  gmsh::vectorpair dimtags_2d;
//  double const x = center[0];
//  double const y = center[1];
//  double const z = center[2];
//  // Do the innermost disk
//  int const circle0_tag = gmsh::model::occ::addCircle(x, y, z, radii[0]);
//  int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
//  int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
//  dimtags_2d.emplace_back(2, disk_tag);
//  // Do the annuli
//  int prev_loop_tag = loop0_tag;
//  for (size_t i = 1; i < nradii; ++i) {
//    int const circle_tag = gmsh::model::occ::addCircle(x, y, z, radii[i]);
//    int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
//    int const annulus_tag = gmsh::model::occ::addPlaneSurface({loop_tag,
//    prev_loop_tag}); dimtags_2d.emplace_back(2, annulus_tag); prev_loop_tag = loop_tag;
//  }
//  gmsh::vectorpair out_dimtags;
//  // NOLINTNEXTLINE(readability-suspicious-call-argument) justification: this is wrong
//  gmsh::model::occ::extrude(dimtags_2d, 0.0, 0.0, height, out_dimtags);
//  gmsh::model::occ::synchronize();
//  std::vector<int> out_tags;
//  out_tags.reserve(nradii);
//  for (auto const & dt : out_dimtags) {
//    if (dt.first == 3) {
//      out_tags.emplace_back(dt.second);
//    }
//  }
//  // Add materials
//  for (size_t i = 0; i < nradii; ++i) {
//    addToPhysicalGroup(3, {out_tags[i]}, -1,
//                       "Material_" + std::string(materials[i].getName().c_str()));
//    // Color entities according to materials
//    Color const color = materials[i].getColor();
//    gmsh::model::setColor(
//        {
//            {3, out_tags[i]}
//    }, // Entities to color
//        static_cast<int>(color.r()), static_cast<int>(color.g()),
//        static_cast<int>(color.b()), static_cast<int>(color.a()),
//        /*recursive=*/true);
//  }
//  return out_tags;
//} // addCylindricalPin
//
// auto
//// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// addCylindricalPinLattice(std::vector<std::vector<double>> const & radii,
//                          std::vector<std::vector<Material>> const & materials,
//                          double const height, std::vector<Vec2d> const & dxdy,
//                          std::vector<std::vector<int>> const & pin_ids,
//                          Vec3d const & offset) -> std::vector<int>
//{
//   LOG_INFO("Adding 2D cylindrical pin lattice");
//   // Input checking
//   size_t const nunique_pins = radii.size();
//   if (nunique_pins == 0) {
//     LOG_ERROR("radii must not be empty");
//   }
//   if (nunique_pins != materials.size()) {
//     LOG_ERROR("Number of radii vectors must be equal to number of material vectors");
//   }
//   if (nunique_pins != dxdy.size()) {
//     LOG_ERROR("Number of radii vectors must be equal to number of dxdy pairs");
//   }
//   if (height <= 0.0) {
//     LOG_ERROR("height must be positive");
//   }
//   // Ensure each row of pin_ids has the same length
//   size_t const ncol = pin_ids[0].size();
//   for (auto const & row : pin_ids) {
//     if (row.size() != ncol) {
//       LOG_ERROR("Each row of pin_ids must have the same length");
//     }
//   }
//   // Ensure all pin_ids are in range 0:nunique_pins - 1.
//   for (auto const & row : pin_ids) {
//     for (auto const & id : row) {
//       if (id < 0 || static_cast<size_t>(id) >= nunique_pins) {
//         LOG_ERROR("pin_ids must be in range [0, nunique_pins - 1]");
//       }
//     }
//   }
//   size_t const nrow = pin_ids.size();
//   size_t const npins = nrow * ncol;
//   // Reverse the ordering of the rows of pin_ids, so the vector ordering matches the
//   // spatial ordering.
//   std::vector<std::vector<int>> pin_ids_rev(nrow);
//   for (size_t i = 0; i < nrow; ++i) {
//     pin_ids_rev[i] = pin_ids[nrow - i - 1];
//   }
//   // Construct a RectilinearGrid object using the vector of AxisAlignedBox constructor.
//   Vector<AxisAlignedBox2F> boxes(static_cast<Int>(npins));
//   double ymin = 0.0;
//   for (size_t i = 0; i < nrow; ++i) {
//     double xmin = 0.0;
//     std::vector<int> const & row = pin_ids_rev[i];
//     auto pin_idx = static_cast<size_t>(row[0]);
//     double const ymax = ymin + dxdy[pin_idx][1];
//     for (size_t j = 0; j < ncol; ++j) {
//       pin_idx = static_cast<size_t>(row[j]);
//       double const xmax = xmin + dxdy[pin_idx][0];
//       boxes[static_cast<Int>(i * ncol + j)] =
//           AxisAlignedBox2F(Point2F(castIfNot<Float>(xmin), castIfNot<Float>(ymin)),
//                           Point2F(castIfNot<Float>(xmax), castIfNot<Float>(ymax)));
//       xmin = xmax;
//     }
//     ymin = ymax;
//   }
//
//   RectilinearGrid2F const grid(boxes);
//
//   std::vector<int> out_tags;
//   // For each unique pin, loop through the pin_ids_rev array and add the pins.
//   for (size_t pin_id = 0; pin_id < nunique_pins; ++pin_id) {
//     size_t const nrad = radii[pin_id].size();
//     if (nrad == 0) {
//       continue;
//     }
//     std::vector<std::vector<int>> material_ids(nrad);
//     for (size_t i = 0; i < nrow; ++i) {
//       std::vector<int> const & row = pin_ids_rev[i];
//       for (size_t j = 0; j < ncol; ++j) {
//         if (row[j] == static_cast<int>(pin_id)) {
//           AxisAlignedBox2F const box = grid.getBox(j, i);
//           double const x = 0.5 * castIfNot<double>(box.xMin() + box.xMax()) +
//           offset[0]; double const y = 0.5 * castIfNot<double>(box.yMin() + box.yMax())
//           + offset[1]; double const z = offset[2];
//           // Do the innermost disk
//           double const r0 = radii[pin_id][0];
//           int const circle0_tag = gmsh::model::occ::addCircle(x, y, z, r0);
//           int const loop0_tag = gmsh::model::occ::addCurveLoop({circle0_tag});
//           int const disk_tag = gmsh::model::occ::addPlaneSurface({loop0_tag});
//           material_ids[0].emplace_back(disk_tag);
//           out_tags.emplace_back(disk_tag);
//           // Do the annuli
//           int prev_loop_tag = loop0_tag;
//           for (size_t k = 1; k < nrad; ++k) {
//             double const r = radii[pin_id][k];
//             int const circle_tag = gmsh::model::occ::addCircle(x, y, z, r);
//             int const loop_tag = gmsh::model::occ::addCurveLoop({circle_tag});
//             int const annulus_tag =
//                 gmsh::model::occ::addPlaneSurface({loop_tag, prev_loop_tag});
//             material_ids[k].emplace_back(annulus_tag);
//             out_tags.emplace_back(annulus_tag);
//             prev_loop_tag = loop_tag;
//           }
//         }
//       }
//     }
//     gmsh::model::occ::synchronize();
//     // Extrude and add materials
//     for (size_t i = 0; i < nrad; ++i) {
//       size_t const nents = material_ids[i].size();
//       if (nents == 0) {
//         continue;
//       }
//       gmsh::vectorpair out_dimtags;
//       gmsh::vectorpair in_dimtags(nents);
//       for (size_t j = 0; j < nents; ++j) {
//         in_dimtags[j] = std::make_pair(2, material_ids[i][j]);
//       }
//       gmsh::model::occ::extrude(in_dimtags, 0.0, 0.0, height, out_dimtags);
//       gmsh::model::occ::synchronize();
//       // Overwrite the material ids with the extruded tags.
//       size_t ctr = 0;
//       for (auto const & dimtag : out_dimtags) {
//         if (dimtag.first == 3) {
//           material_ids[i][ctr] = dimtag.second;
//           out_tags.emplace_back(dimtag.second);
//           ++ctr;
//         }
//       }
//       ASSERT(ctr == nents);
//       addToPhysicalGroup(
//           3,                                                                // dim
//           material_ids[i],                                                  // tags
//           -1,                                                               // tag
//           "Material_" + std::string(materials[pin_id][i].getName().c_str()) // name
//       );
//       // Color entities according to materials
//       gmsh::vectorpair mat_dimtags(nents);
//       for (size_t j = 0; j < nents; ++j) {
//         mat_dimtags[j] = std::make_pair(3, material_ids[i][j]);
//       }
//       Color const color = materials[pin_id][i].getColor();
//       gmsh::model::setColor(mat_dimtags, static_cast<int>(color.r()),
//                             static_cast<int>(color.g()), static_cast<int>(color.b()),
//                             static_cast<int>(color.a()), /*recursive=*/true);
//     }
//   } // end loop over unique pins
//   return out_tags;
// } // end function
//
//==============================================================================
//  overlayCoarseGrid
//==============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
overlayCoarseGrid(mpact::Model const & model, Material const & fill_material)
{
  LOG_INFO("Overlaying MPACT coarse grid");
  // Algorithm:
  //  1. Get the lower left corner of each unique 2D coarse cell rectangle that
  //      will cut the model.
  //  2. Create the rectangles. Since group preserving fragment only keeps the
  //      highest dimension groups, if the model is 3D, we create a box whose
  //      bottom face is the coarse cell rectangle.
  //  3. Assign the fill material and coarse cell labels to these entities.
  //  4. Group preserving fragment with the model and grid.
  //  4a. If the model is 3D, group preserving intersection between the model
  //      and the 2D rectangles.
  //  5. Remove all entities that are not coarse cells.

  // Get all model entities prior to adding the grid
  gmsh::vectorpair model_dimtags;
  gmsh::model::getEntities(model_dimtags, 3);
  // Determine model dimension
  int model_dim = 3;
  if (model_dimtags.empty()) {
    gmsh::model::getEntities(model_dimtags, 2);
    model_dim = 2;
  }
  // Get the unique coarse cell lower left corners
  Int const num_cc = model.numCoarseCells();
  Vector<Point3F> cc_lower_lefts(num_cc); // Of the cut-plane
  Vector<Vec3F> cc_extents(num_cc);
  Vector<int8_t> cc_found(num_cc, 0);
  Vector<int8_t> rtm_found(model.numRTMs(), 0);
  Vector<int8_t> lat_found(model.numLattices(), 0);
  Vector<int8_t> asy_found(model.numAssemblies(), 0);

  auto const & core = model.getCore();
  if (core.children().empty()) {
    LOG_ERROR("Core has no children");
    return;
  }
  // For each assembly
  Int const nyasy = core.grid().numCells(1);
  Int const nxasy = core.grid().numCells(0);
  for (Int iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Int ixasy = 0; ixasy < nxasy; ++ixasy) {
      auto const asy_id = core.getChild(ixasy, iyasy);
      if (asy_found[asy_id] == 1) {
        continue;
      }
      asy_found[asy_id] = 1;
      AxisAlignedBox2F const asy_bb = core.grid().getBox(ixasy, iyasy);
      Point2F const asy_ll = asy_bb.minima(); // Lower left corner
      auto const & assembly = model.getAssembly(asy_id);
      if (assembly.children().empty()) {
        LOG_ERROR("Assembly has no children");
        return;
      }
      // For each lattice
      Int const nzlat = assembly.grid().numCells(0);
      for (Int izlat = 0; izlat < nzlat; ++izlat) {
        auto const lat_id = assembly.getChild(izlat);
        if (lat_found[lat_id] == 1) {
          continue;
        }
        lat_found[lat_id] = 1;
        auto const low_z = castIfNot<double>(assembly.grid().divs(0)[izlat]);
        auto const high_z = castIfNot<double>(assembly.grid().divs(0)[izlat + 1]);
        double const z_cut = (low_z + high_z) / 2;
        // Only half the thickness, since we want to cut at the midpoint
        double const dz = (high_z - low_z) / 2;
        auto const & lattice = model.getLattice(lat_id);
        if (lattice.children().empty()) {
          LOG_ERROR("Lattice has no children");
          return;
        }
        // For each RTM
        Int const nyrtm = lattice.grid().numCells(1);
        Int const nxrtm = lattice.grid().numCells(0);
        for (Int iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Int ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
            auto const rtm_id = lattice.getChild(ixrtm, iyrtm);
            if (rtm_found[rtm_id] == 1) {
              continue;
            }
            rtm_found[rtm_id] = 1;
            AxisAlignedBox2F const rtm_bb = lattice.grid().getBox(ixrtm, iyrtm);
            Point2F const rtm_ll = rtm_bb.minima(); // Lower left corner
            auto const & rtm = model.getRTM(rtm_id);
            if (rtm.children().empty()) {
              LOG_ERROR("RTM has no children");
              return;
            }
            // For each coarse cell
            Int const nycells = rtm.grid().numCells(1);
            Int const nxcells = rtm.grid().numCells(0);
            for (Int iycell = 0; iycell < nycells; ++iycell) {
              for (Int ixcell = 0; ixcell < nxcells; ++ixcell) {
                auto const cell_id = rtm.getChild(ixcell, iycell);
                if (cc_found[cell_id] == 1) {
                  continue;
                }
                cc_found[cell_id] = 1;
                AxisAlignedBox2F const cell_bb = rtm.grid().getBox(ixcell, iycell);
                Point2F const ll = asy_ll + rtm_ll + cell_bb.minima();
                cc_lower_lefts[cell_id][0] = ll[0];
                cc_lower_lefts[cell_id][1] = ll[1];
                cc_lower_lefts[cell_id][2] = castIfNot<Float>(z_cut);
                cc_extents[cell_id][0] = cell_bb.extents(0);
                cc_extents[cell_id][1] = cell_bb.extents(1);
                cc_extents[cell_id][2] = castIfNot<Float>(dz);
              } // cell
            } // cell
          } // rtm
        } // rtm
      } // lat
    } // assembly
  } // assembly

  // Get materials and see if the fill material already exists
  // If it does, move it to the end of the material hierarchy, otherwise
  // append it to the end.
  ASSERT(!fill_material.getName().empty());
  std::string const fill_material_name(fill_material.getName().data());
  Color const fill_material_color = fill_material.getColor();
  std::vector<Material> materials;
  um2::gmsh::model::getMaterials(materials);
  bool fill_exists = false;
  size_t const num_materials = materials.size();
  for (size_t i = 0; i < num_materials; ++i) {
    std::string const name(materials[i].getName().data());
    if (name == fill_material_name) {
      // Move the material to the end of the list to ensure
      // it is the fill material
      std::swap(materials[i], materials[num_materials - 1]);
      fill_exists = true;
      LOG_DEBUG("Found fill material: ", fill_material_name.c_str());
      break;
    }
  }
  if (!fill_exists) {
    Material tmp_mat;
    tmp_mat.setName(String(fill_material_name.c_str()));
    tmp_mat.setColor(fill_material_color);
    materials.emplace_back(um2::move(tmp_mat));
  }
  std::vector<int> cc_tags(static_cast<size_t>(num_cc));
  // Potential for float promotion here
  namespace factory = gmsh::model::occ;
  if (model_dim == 2) {
    // Create rectangles
    for (Int i = 0; i < num_cc; ++i) {
      auto const & ll = cc_lower_lefts[i];
      auto const & ext = cc_extents[i];
      cc_tags[static_cast<size_t>(i)] = factory::addRectangle(
          castIfNot<double>(ll[0]), castIfNot<double>(ll[1]), castIfNot<double>(ll[2]),
          castIfNot<double>(ext[0]), castIfNot<double>(ext[1]));
    }
  } else {
    // Create boxes
    for (Int i = 0; i < num_cc; ++i) {
      auto const & ll = cc_lower_lefts[i];
      auto const & ext = cc_extents[i];
      cc_tags[static_cast<size_t>(i)] =
          factory::addBox(castIfNot<double>(ll[0]), castIfNot<double>(ll[1]),
                          castIfNot<double>(ll[2]), castIfNot<double>(ext[0]),
                          castIfNot<double>(ext[1]), castIfNot<double>(ext[2]));
    }
  }
  factory::synchronize();

  // Assign physical groups
  // Fill material is assigned to all slice rectangles
  std::string const fill_mat_full_name = "Material_" + fill_material_name;
  addToPhysicalGroup(model_dim, cc_tags, -1, fill_mat_full_name);

  // Add a physical group for each coarse cell
  std::string coarse_cell_name("Coarse_Cell_00000");
  for (Int i = 0; i < num_cc; ++i) {
    gmsh::model::addPhysicalGroup(model_dim, {cc_tags[static_cast<size_t>(i)]}, -1,
                                  coarse_cell_name);
    mpact::incrementASCIINumber(coarse_cell_name);
  }

  // Fragment
  gmsh::vectorpair grid_dimtags(cc_tags.size());
  for (size_t i = 0; i < cc_tags.size(); ++i) {
    grid_dimtags[i] = {model_dim, cc_tags[i]};
  }
  std::sort(grid_dimtags.begin(), grid_dimtags.end());
  gmsh::vectorpair out_dimtags;
  std::vector<gmsh::vectorpair> out_dimtags_map;
  groupPreservingFragment(model_dimtags, grid_dimtags, out_dimtags, out_dimtags_map,
                          materials);

  // Remove all entities that do not have a "Coarse_Cell" physical group
  // associated with them.
  std::vector<int> dont_remove;
  gmsh::vectorpair group_dimtags;
  gmsh::model::getPhysicalGroups(group_dimtags, model_dim);
  for (auto const & dimtag : group_dimtags) {
    int const dim = dimtag.first;
    ASSERT(dim == model_dim);
    int const tag = dimtag.second;
    std::string name;
    // We do not expect entities to have many physical groups. Therefore,
    // we insert the whole list of entities for each physical group into
    // the to_remove vector.
    gmsh::model::getPhysicalName(dim, tag, name);
    if (name.starts_with("Coarse_Cell")) {
      std::vector<int> tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags);
      dont_remove.insert(dont_remove.end(), tags.begin(), tags.end());
    }
  }
  std::sort(dont_remove.begin(), dont_remove.end());
  gmsh::vectorpair dont_remove_dimtags;
  if (!dont_remove.empty()) {
    dont_remove_dimtags.emplace_back(model_dim, dont_remove[0]);
  }
  for (size_t i = 1; i < dont_remove.size(); ++i) {
    if (dont_remove[i] != dont_remove[i - 1]) {
      dont_remove_dimtags.emplace_back(model_dim, dont_remove[i]);
    }
  }
  LOG_INFO("Removing unneeded entities");
  gmsh::vectorpair to_remove_dimtags;
  gmsh::vectorpair all_dimtags;
  gmsh::model::getEntities(
      all_dimtags, model_dim); // This was changed in the refactor to include model_dim
  std::set_difference(all_dimtags.begin(), all_dimtags.end(), dont_remove_dimtags.begin(),
                      dont_remove_dimtags.end(), std::back_inserter(to_remove_dimtags));
  gmsh::model::occ::remove(to_remove_dimtags, /*recursive=*/true);
  gmsh::model::removeEntities(to_remove_dimtags, /*recursive=*/true);
  gmsh::model::occ::synchronize();
  if (model_dim == 3) {
    // We need to create a 2D version of each physical group, then intersect the 2D
    // grid.
    group_dimtags.clear();
    gmsh::model::getPhysicalGroups(group_dimtags);
    for (auto const & dimtag : group_dimtags) {
      int const dim = dimtag.first;
      ASSERT(dim == 3);
      int const tag = dimtag.second;
      std::string name;
      gmsh::model::getPhysicalName(dim, tag, name);
      std::vector<int> tags_3d;
      gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags_3d);
      std::vector<int> tags_2d;
      for (auto const & etag : tags_3d) {
        std::vector<int> upward;
        std::vector<int> downward;
        gmsh::model::getAdjacencies(3, etag, upward, downward);
        // If the downard tags are not in the list of 2D entities, then add them.
        // We expect many repeated tags, so we use a binary search to find the
        // insertion point.
        for (auto const & tag_2 : downward) {
          auto it = std::lower_bound(tags_2d.begin(), tags_2d.end(), tag_2);
          if (it == tags_2d.end() || *it != tag_2) {
            tags_2d.insert(it, tag_2);
          }
        }
      }
      // Create a new physical group for the 2D entities
      gmsh::model::addPhysicalGroup(2, tags_2d, -1, name);
    }
    gmsh::vectorpair model_dimtags_2d;
    gmsh::model::getEntities(model_dimtags_2d, 2);
    std::vector<int> cc_tags_2d(static_cast<size_t>(num_cc));
    // Create rectangles
    for (Int i = 0; i < num_cc; ++i) {
      auto const & ll = cc_lower_lefts[i];
      auto const & ext = cc_extents[i];
      cc_tags_2d[static_cast<size_t>(i)] = factory::addRectangle(
          castIfNot<double>(ll[0]), castIfNot<double>(ll[1]), castIfNot<double>(ll[2]),
          castIfNot<double>(ext[0]), castIfNot<double>(ext[1]));
    }
    factory::synchronize();
    // Don't need to add coarse cell physical groups to the 2D grid. The model
    // already has the physical groups.
    gmsh::vectorpair grid_dimtags_2d(cc_tags_2d.size());
    for (size_t i = 0; i < cc_tags_2d.size(); ++i) {
      grid_dimtags_2d[i] = {2, cc_tags_2d[i]};
    }
    std::sort(grid_dimtags_2d.begin(), grid_dimtags_2d.end());
    // remove all 3D entities (leaving surfaces), then intersect the 2D grid
    to_remove_dimtags.clear();
    gmsh::model::getEntities(to_remove_dimtags, 3);
    gmsh::model::removeEntities(to_remove_dimtags);
    gmsh::model::occ::remove(to_remove_dimtags);
    gmsh::vectorpair out_dimtags_2d;
    std::vector<gmsh::vectorpair> out_dimtags_map_2d;
    groupPreservingIntersect(model_dimtags_2d, grid_dimtags_2d, out_dimtags_2d,
                             out_dimtags_map_2d, materials);
    // Remove all entities that are not children of the grid entities.
    std::vector<int> dont_remove_2d;
    size_t const nmodel_2d = model_dimtags_2d.size();
    size_t const ngrid_2d = grid_dimtags_2d.size();
    for (size_t i = nmodel_2d; i < nmodel_2d + ngrid_2d; ++i) {
      for (auto const & child : out_dimtags_map_2d[i]) {
        auto child_it =
            std::lower_bound(dont_remove_2d.begin(), dont_remove_2d.end(), child.second);
        if (child_it == dont_remove_2d.end() || *child_it != child.second) {
          dont_remove_2d.insert(child_it, child.second);
        }
      }
    }
    gmsh::vectorpair dont_remove_dimtags_2d(dont_remove_2d.size());
    for (size_t i = 0; i < dont_remove_2d.size(); ++i) {
      dont_remove_dimtags_2d[i] = {2, dont_remove_2d[i]};
    }
    gmsh::vectorpair to_remove_dimtags_2d;
    gmsh::vectorpair all_dimtags_2d;
    gmsh::model::getEntities(all_dimtags_2d, 2);
    std::set_difference(all_dimtags_2d.begin(), all_dimtags_2d.end(),
                        dont_remove_dimtags_2d.begin(), dont_remove_dimtags_2d.end(),
                        std::back_inserter(to_remove_dimtags_2d));
    gmsh::vectorpair all_dimtags_3d;
    gmsh::model::getEntities(all_dimtags_3d, 3);
    gmsh::model::occ::remove(all_dimtags_3d);
    gmsh::model::removeEntities(all_dimtags_3d);
    // gmsh::model::occ::remove(to_remove_dimtags_2d, true); Already removed via intersect
    gmsh::model::removeEntities(to_remove_dimtags_2d, /*recursive=*/true);
    gmsh::model::occ::synchronize();
  }
}

} // namespace occ
} // namespace um2::gmsh::model
#endif // UM2_USE_GMSH
