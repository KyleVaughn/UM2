#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <cstddef>
#  include <string>
#  include <um2/common/color.hpp>
#  include <um2/common/logger.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/model.hpp>
#  include <um2/physics/material.hpp>
#  include <vector>
#endif

#include "../test_macros.hpp"

#if UM2_USE_GMSH

TEST_CASE(addToPhysicalGroup)
{
  um2::gmsh::initialize();
  um2::gmsh::model::occ::addDisk(0.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::addDisk(2.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, {1}, -1, "disks");
  um2::gmsh::model::addToPhysicalGroup(2, {2}, -1, "disks");
  std::vector<int> tags;
  um2::gmsh::model::getEntitiesForPhysicalGroup(2, 1, tags);
  ASSERT(tags.size() == 2);
  ASSERT(tags[0] == 1);
  ASSERT(tags[1] == 2);
  um2::gmsh::vectorpair dimtags;
  um2::gmsh::model::getPhysicalGroups(dimtags);
  ASSERT(dimtags.size() == 1);
  um2::gmsh::finalize();
}

TEST_CASE(getMaterials)
{
  um2::gmsh::initialize();
  um2::gmsh::model::occ::addDisk(0.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::addDisk(2.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::addDisk(4.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, {1}, -1, "Material_UO2");
  um2::gmsh::model::setColor(
      {
          {2, 1}
  },
      255, 0, 0);
  um2::gmsh::model::addPhysicalGroup(2, {2, 3}, -1, "Material_MOX");
  um2::gmsh::model::setColor(
      {
          {2, 2},
          {2, 3}
  },
      0, 0, 255);
  std::vector<um2::Material> materials;
  um2::gmsh::model::getMaterials(materials);
  ASSERT(materials.size() == 2);
  ASSERT(materials[0].getName() == "UO2");
  ASSERT(materials[0].getColor() == um2::red);
  ASSERT(materials[1].getName() == "MOX");
  ASSERT(materials[1].getColor() == um2::blue);
  um2::gmsh::finalize();
}

TEST_CASE(groupPresFragment_2d2d_basic)
{
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setColor(um2::red);
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::blue);
  std::vector<um2::Material> const materials = {fuel, moderator};
  // First pass no material hierarchy, second pass with material hierarchy
  for (size_t i = 0; i < 2; ++i) {
    um2::gmsh::initialize();
    int const tag1 = um2::gmsh::model::occ::addDisk(-1, 0, 0, 2, 2);
    int const tag2 = um2::gmsh::model::occ::addDisk(1, 0, 0, 2, 2);
    um2::gmsh::model::occ::synchronize();
    int const group_tag1 = um2::gmsh::model::addPhysicalGroup(2, {tag1, tag2}, -1, "A");
    int const mat_tag1 =
        um2::gmsh::model::addPhysicalGroup(2, {tag1}, -1, "Material_Fuel");
    int const group_tag2 = um2::gmsh::model::addPhysicalGroup(2, {tag2}, -1, "B");
    int const mat_tag2 =
        um2::gmsh::model::addPhysicalGroup(2, {tag2}, -1, "Material_Moderator");
    um2::gmsh::vectorpair const object_dimtags = {
        {2, tag1}
    };
    um2::gmsh::vectorpair const tool_dimtags = {
        {2, tag2}
    };
    um2::gmsh::vectorpair out_dimtags;
    std::vector<um2::gmsh::vectorpair> out_dimtags_map;
    // Fragment into a venn diagram. After the fragment:
    // Left entity is entity 2, middle entity is entity 1, right entity is entity 3.
    if (i == 0) {
      um2::gmsh::model::occ::groupPreservingFragment(object_dimtags, tool_dimtags,
                                                     out_dimtags, out_dimtags_map);
    } else {
      um2::gmsh::model::occ::groupPreservingFragment(
          object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map, materials);
    }
    // out_dimtags should have 3 entities: {{2, 1}, {2, 2}, {2, 3}}
    ASSERT(out_dimtags.size() == 3);
    ASSERT(out_dimtags[0].first == 2);
    ASSERT(out_dimtags[0].second == 1);
    ASSERT(out_dimtags[1].first == 2);
    ASSERT(out_dimtags[1].second == 2);
    ASSERT(out_dimtags[2].first == 2);
    ASSERT(out_dimtags[2].second == 3);
    // out_dimtags_map should have 2 entities: {{{2, 1}, {2, 2}}, {{2, 3}, {2, 1}}}
    ASSERT(out_dimtags_map.size() == 2);
    ASSERT(out_dimtags_map[0].size() == 2);
    ASSERT(out_dimtags_map[0][0].first == 2);
    ASSERT(out_dimtags_map[0][0].second == 1);
    ASSERT(out_dimtags_map[0][1].first == 2);
    ASSERT(out_dimtags_map[0][1].second == 2);
    ASSERT(out_dimtags_map[1].size() == 2);
    ASSERT(out_dimtags_map[1][0].first == 2);
    ASSERT(out_dimtags_map[1][0].second == 3);
    ASSERT(out_dimtags_map[1][1].first == 2);
    ASSERT(out_dimtags_map[1][1].second == 1);
    // Group A should have 3 entities: {1, 2, 3}
    std::vector<int> tags;
    std::string name;
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, group_tag1, tags);
    um2::gmsh::model::getPhysicalName(2, group_tag1, name);
    ASSERT(name == "A");
    ASSERT(tags.size() == 3);
    ASSERT(tags[0] == 1);
    ASSERT(tags[1] == 2);
    ASSERT(tags[2] == 3);
    // Group B should have 2 entities: {1, 3}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, group_tag2, tags);
    um2::gmsh::model::getPhysicalName(2, group_tag2, name);
    ASSERT(name == "B");
    ASSERT(tags.size() == 2);
    ASSERT(tags[0] == 1);
    ASSERT(tags[1] == 3);
    // Material Fuel should have 2 entities: {1, 2}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, mat_tag1, tags);
    um2::gmsh::model::getPhysicalName(2, mat_tag1, name);
    ASSERT(name == "Material_Fuel");
    ASSERT(tags.size() == 2);
    ASSERT(tags[0] == 1);
    ASSERT(tags[1] == 2);
    if (i == 1) {
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(2, 1, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() &&
             a == um2::red.a());
      um2::gmsh::model::getColor(2, 2, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() &&
             a == um2::red.a());
    }
    // if material hierarchy is used, Material Moderator should have 1 entity: {3}
    // if material hierarchy is not used, Material Moderator should have 2 entities:
    // {1, 3}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, mat_tag2, tags);
    um2::gmsh::model::getPhysicalName(2, mat_tag2, name);
    ASSERT(name == "Material_Moderator");
    if (i == 0) {
      ASSERT(tags.size() == 2);
      ASSERT(tags[0] == 1);
      ASSERT(tags[1] == 3);
    } else {
      ASSERT(tags.size() == 1);
      ASSERT(tags[0] == 3);
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(2, 3, r, g, b, a);
      ASSERT(r == um2::blue.r() && g == um2::blue.g() && b == um2::blue.b() &&
             a == um2::blue.a());
    }
    um2::gmsh::finalize();
  }
}

TEST_CASE(groupPresFragment_2d2d_complex)
{
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setColor(um2::red);
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::blue);
  std::vector<um2::Material> const materials = {fuel, moderator};
  um2::gmsh::initialize();
  int const disk_tag = um2::gmsh::model::occ::addDisk(0, 0, 0, 1, 1);
  std::vector<int> grid_tags(4);
  grid_tags[0] = um2::gmsh::model::occ::addRectangle(-2, -2, 0, 2, 2);
  grid_tags[1] = um2::gmsh::model::occ::addRectangle(0, -2, 0, 2, 2);
  grid_tags[2] = um2::gmsh::model::occ::addRectangle(-2, 0, 0, 2, 2);
  grid_tags[3] = um2::gmsh::model::occ::addRectangle(0, 0, 0, 2, 2);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, grid_tags, -1, "Grid");
  um2::gmsh::model::addPhysicalGroup(2, {disk_tag}, -1, "Disk");
  int const fuel_ptag =
      um2::gmsh::model::addPhysicalGroup(2, {disk_tag}, -1, "Material_Fuel");
  int const moderator_ptag =
      um2::gmsh::model::addPhysicalGroup(2, grid_tags, -1, "Material_Moderator");
  um2::gmsh::vectorpair const object_dimtags = {
      {2, disk_tag}
  };
  um2::gmsh::vectorpair const tool_dimtags = {
      {2, grid_tags[0]},
      {2, grid_tags[1]},
      {2, grid_tags[2]},
      {2, grid_tags[3]}
  };
  um2::gmsh::vectorpair out_dimtags;
  std::vector<um2::gmsh::vectorpair> out_dimtags_map;
  um2::gmsh::model::occ::groupPreservingFragment(object_dimtags, tool_dimtags,
                                                 out_dimtags, out_dimtags_map, materials);
  // Group Fuel should have 4 entities: {1, 2, 3, 4}
  std::vector<int> tags;
  std::string name;
  um2::gmsh::model::getEntitiesForPhysicalGroup(2, fuel_ptag, tags);
  um2::gmsh::model::getPhysicalName(2, fuel_ptag, name);
  ASSERT(name == "Material_Fuel");
  ASSERT(tags.size() == 4);
  ASSERT(tags[0] == 1);
  ASSERT(tags[1] == 2);
  ASSERT(tags[2] == 3);
  ASSERT(tags[3] == 4);
  // Group Moderator should have 4 entities: {5, 6, 7, 8}
  tags.clear();
  um2::gmsh::model::getEntitiesForPhysicalGroup(2, moderator_ptag, tags);
  um2::gmsh::model::getPhysicalName(2, moderator_ptag, name);
  ASSERT(name == "Material_Moderator");
  ASSERT(tags.size() == 4);
  ASSERT(tags[0] == 5);
  ASSERT(tags[1] == 6);
  ASSERT(tags[2] == 7);
  ASSERT(tags[3] == 8);
  um2::gmsh::finalize();
}

TEST_CASE(groupPresFragment_3d3d)
{
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setColor(um2::red);
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::blue);
  std::vector<um2::Material> const materials = {fuel, moderator};
  // First pass no material hierarchy, second pass with material hierarchy
  for (size_t i = 0; i < 2; ++i) {
    um2::gmsh::initialize();
    int const tag1 = um2::gmsh::model::occ::addSphere(-1, 0, 0, 2);
    int const tag2 = um2::gmsh::model::occ::addSphere(1, 0, 0, 2);
    um2::gmsh::model::occ::synchronize();
    int const group_tag1 = um2::gmsh::model::addPhysicalGroup(3, {tag1, tag2}, -1, "A");
    int const mat_tag1 =
        um2::gmsh::model::addPhysicalGroup(3, {tag1}, -1, "Material_Fuel");
    int const group_tag2 = um2::gmsh::model::addPhysicalGroup(3, {tag2}, -1, "B");
    int const mat_tag2 =
        um2::gmsh::model::addPhysicalGroup(3, {tag2}, -1, "Material_Moderator");
    um2::gmsh::vectorpair const object_dimtags = {
        {3, tag1}
    };
    um2::gmsh::vectorpair const tool_dimtags = {
        {3, tag2}
    };
    um2::gmsh::vectorpair out_dimtags;
    std::vector<um2::gmsh::vectorpair> out_dimtags_map;
    // Fragment into a venn diagram. After the fragment:
    // Left entity is entity 2, middle entity is entity 1, right entity is entity 3.
    if (i == 0) {
      um2::gmsh::model::occ::groupPreservingFragment(object_dimtags, tool_dimtags,
                                                     out_dimtags, out_dimtags_map);
    } else {
      um2::gmsh::model::occ::groupPreservingFragment(
          object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map, materials);
    }
    // out_dimtags should have 3 entities: {{3, 1}, {3, 2}, {3, 3}}
    ASSERT(out_dimtags.size() == 3);
    ASSERT(out_dimtags[0].first == 3);
    ASSERT(out_dimtags[0].second == 1);
    ASSERT(out_dimtags[1].first == 3);
    ASSERT(out_dimtags[1].second == 2);
    ASSERT(out_dimtags[2].first == 3);
    ASSERT(out_dimtags[2].second == 3);
    // out_dimtags_map should have 2 entities: {{{3, 1}, {3, 2}}, {{3, 3}, {3, 2}}}
    ASSERT(out_dimtags_map.size() == 2);
    ASSERT(out_dimtags_map[0].size() == 2);
    ASSERT(out_dimtags_map[0][0].first == 3);
    ASSERT(out_dimtags_map[0][0].second == 1);
    ASSERT(out_dimtags_map[0][1].first == 3);
    ASSERT(out_dimtags_map[0][1].second == 2);
    ASSERT(out_dimtags_map[1].size() == 2);
    ASSERT(out_dimtags_map[1][0].first == 3);
    ASSERT(out_dimtags_map[1][0].second == 3);
    ASSERT(out_dimtags_map[1][1].first == 3);
    ASSERT(out_dimtags_map[1][1].second == 2);
    // Group A should have 3 entities: {1, 2, 3}
    std::vector<int> tags;
    std::string name;
    um2::gmsh::model::getEntitiesForPhysicalGroup(3, group_tag1, tags);
    um2::gmsh::model::getPhysicalName(3, group_tag1, name);
    ASSERT(name == "A");
    ASSERT(tags.size() == 3);
    ASSERT(tags[0] == 1);
    ASSERT(tags[1] == 2);
    ASSERT(tags[2] == 3);
    // Group B should have 2 entities: {2, 3}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(3, group_tag2, tags);
    um2::gmsh::model::getPhysicalName(3, group_tag2, name);
    ASSERT(name == "B");
    ASSERT(tags.size() == 2);
    ASSERT(tags[0] == 2);
    ASSERT(tags[1] == 3);
    // Material Fuel should have 2 entities: {1, 2}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(3, mat_tag1, tags);
    um2::gmsh::model::getPhysicalName(3, mat_tag1, name);
    ASSERT(name == "Material_Fuel");
    ASSERT(tags.size() == 2);
    ASSERT(tags[0] == 1);
    ASSERT(tags[1] == 2);
    if (i == 1) {
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(3, 1, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() &&
             a == um2::red.a());
      um2::gmsh::model::getColor(3, 2, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() &&
             a == um2::red.a());
    }
    // if material hierarchy is used, Material Moderator should have 1 entity: {3}
    // if material hierarchy is not used, Material Moderator should have 2 entities:
    // {2, 3}
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(3, mat_tag2, tags);
    um2::gmsh::model::getPhysicalName(3, mat_tag2, name);
    ASSERT(name == "Material_Moderator");
    if (i == 0) {
      ASSERT(tags.size() == 2);
      ASSERT(tags[0] == 2);
      ASSERT(tags[1] == 3);
    } else {
      ASSERT(tags.size() == 1);
      ASSERT(tags[0] == 3);
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(3, 3, r, g, b, a);
      ASSERT(r == um2::blue.r() && g == um2::blue.g() && b == um2::blue.b() &&
             a == um2::blue.a());
    }
    um2::gmsh::finalize();
  }
}

TEST_CASE(groupPresIntersect_2d2d_basic)
{
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setColor(um2::red);
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::blue);
  std::vector<um2::Material> const materials = {fuel, moderator};
  // First pass no material hierarchy, second pass with material hierarchy
  for (size_t i = 0; i < 2; ++i) {
    um2::gmsh::initialize();
    int const tag1 = um2::gmsh::model::occ::addDisk(-1, 0, 0, 2, 2);
    int const tag2 = um2::gmsh::model::occ::addDisk(1, 0, 0, 2, 2);
    um2::gmsh::model::occ::synchronize();
    int const group_tag1 = um2::gmsh::model::addPhysicalGroup(2, {tag1, tag2}, -1, "A");
    int const mat_tag1 =
        um2::gmsh::model::addPhysicalGroup(2, {tag1}, -1, "Material_Fuel");
    int const group_tag2 = um2::gmsh::model::addPhysicalGroup(2, {tag2}, -1, "B");
    int const mat_tag2 =
        um2::gmsh::model::addPhysicalGroup(2, {tag2}, -1, "Material_Moderator");
    um2::gmsh::vectorpair const object_dimtags = {
        {2, tag1}
    };
    um2::gmsh::vectorpair const tool_dimtags = {
        {2, tag2}
    };
    um2::gmsh::vectorpair out_dimtags;
    std::vector<um2::gmsh::vectorpair> out_dimtags_map;
    if (i == 0) {
      um2::gmsh::model::occ::groupPreservingIntersect(object_dimtags, tool_dimtags,
                                                      out_dimtags, out_dimtags_map);
    } else {
      um2::gmsh::model::occ::groupPreservingIntersect(
          object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map, materials);
    }
    ASSERT(out_dimtags.size() == 1);
    ASSERT(out_dimtags[0].first == 2);
    ASSERT(out_dimtags[0].second == 1);
    ASSERT(out_dimtags_map.size() == 2);
    ASSERT(out_dimtags_map[0].size() == 1);
    ASSERT(out_dimtags_map[0][0].first == 2);
    ASSERT(out_dimtags_map[0][0].second == 1);
    ASSERT(out_dimtags_map[1].size() == 1);
    ASSERT(out_dimtags_map[1][0].first == 2);
    ASSERT(out_dimtags_map[1][0].second == 1);
    std::vector<int> tags;
    std::string name;
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, group_tag1, tags);
    um2::gmsh::model::getPhysicalName(2, group_tag1, name);
    ASSERT(name == "A");
    ASSERT(tags.size() == 1);
    ASSERT(tags[0] == 1);
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, group_tag2, tags);
    um2::gmsh::model::getPhysicalName(2, group_tag2, name);
    ASSERT(name == "B");
    ASSERT(tags.size() == 1);
    ASSERT(tags[0] == 1);
    tags.clear();
    um2::gmsh::model::getEntitiesForPhysicalGroup(2, mat_tag1, tags);
    um2::gmsh::model::getPhysicalName(2, mat_tag1, name);
    ASSERT(name == "Material_Fuel");
    ASSERT(tags.size() == 1);
    ASSERT(tags[0] == 1);
    tags.clear();
    if (i == 1) {
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(2, 1, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() &&
             a == um2::red.a());
    }
    tags.clear();
    if (i == 0) {
      um2::gmsh::model::getEntitiesForPhysicalGroup(2, mat_tag2, tags);
      um2::gmsh::model::getPhysicalName(2, mat_tag2, name);
      ASSERT(name == "Material_Moderator");
      ASSERT(tags.size() == 1);
      ASSERT(tags[0] == 1);
    }
    um2::gmsh::finalize();
  }
}

TEST_CASE(groupPresIntersect_2d2d_complex)
{
  um2::Material fuel;
  fuel.setName("Fuel");
  fuel.setColor(um2::red);
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setColor(um2::blue);
  std::vector<um2::Material> const materials = {fuel, moderator};
  um2::gmsh::initialize();
  int const disk_tag = um2::gmsh::model::occ::addDisk(0, 0, 0, 1, 1);
  std::vector<int> grid_tags(4);
  grid_tags[0] = um2::gmsh::model::occ::addRectangle(-2, -2, 0, 2, 2);
  grid_tags[1] = um2::gmsh::model::occ::addRectangle(0, -2, 0, 2, 2);
  grid_tags[2] = um2::gmsh::model::occ::addRectangle(-2, 0, 0, 2, 2);
  grid_tags[3] = um2::gmsh::model::occ::addRectangle(0, 0, 0, 2, 2);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, grid_tags, -1, "Grid");
  um2::gmsh::model::addPhysicalGroup(2, {disk_tag}, -1, "Disk");
  um2::gmsh::model::addPhysicalGroup(2, {disk_tag}, -1, "Material_Fuel");
  um2::gmsh::model::addPhysicalGroup(2, grid_tags, -1, "Material_Moderator");
  um2::gmsh::vectorpair const object_dimtags = {
      {2, disk_tag}
  };
  um2::gmsh::vectorpair const tool_dimtags = {
      {2, grid_tags[0]},
      {2, grid_tags[1]},
      {2, grid_tags[2]},
      {2, grid_tags[3]}
  };
  um2::gmsh::vectorpair out_dimtags;
  std::vector<um2::gmsh::vectorpair> out_dimtags_map;
  um2::gmsh::model::occ::groupPreservingIntersect(
      object_dimtags, tool_dimtags, out_dimtags, out_dimtags_map, materials);
  // Ensure that there are only 3 groups
  um2::gmsh::vectorpair dimtags;
  um2::gmsh::model::getPhysicalGroups(dimtags);
  ASSERT(dimtags.size() == 3);
  // Group Fuel should have 4 entities: {5, 6, 7, 8}
  std::vector<int> tags;
  std::string name;
  tags.clear();
  um2::gmsh::model::getEntitiesForPhysicalGroup(2, 3, tags);
  um2::gmsh::model::getPhysicalName(2, 3, name);
  ASSERT(name == "Material_Fuel");
  ASSERT(tags.size() == 4);
  ASSERT(tags[0] == 2);
  ASSERT(tags[1] == 3);
  ASSERT(tags[2] == 4);
  ASSERT(tags[3] == 5);
  um2::gmsh::finalize();
}

TEST_SUITE(gmsh_model)
{
  TEST(addToPhysicalGroup);
  TEST(getMaterials);
  TEST(groupPresFragment_2d2d_basic);
  TEST(groupPresFragment_2d2d_complex);
  TEST(groupPresFragment_3d3d);
  TEST(groupPresIntersect_2d2d_basic);
  TEST(groupPresIntersect_2d2d_complex);
}
#endif // UM2_USE_GMSH

#if !UM2_USE_GMSH
CONST
#endif
auto
main() -> int
{
#if UM2_USE_GMSH
  um2::logger::level = um2::logger::levels::error;
  RUN_SUITE(gmsh_model);
#endif
  return 0;
}
