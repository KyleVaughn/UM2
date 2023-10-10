#include <um2/config.hpp>
#if UM2_USE_GMSH
#  include <um2/gmsh/model.hpp>
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
  ASSERT(materials[0].name == "UO2");
  ASSERT(materials[0].color == um2::red);
  ASSERT(materials[1].name == "MOX");
  ASSERT(materials[1].color == um2::blue);
  um2::gmsh::finalize();
}

TEST_CASE(groupPresFragment_2d2d)
{
  std::vector<um2::Material> const materials = {um2::Material("Fuel", "red"),
                                                um2::Material("Moderator", "blue")};
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
      um2::Color const red("red");
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(2, 1, r, g, b, a);
      ASSERT(r == red.r() && g == red.g() && b == red.b() && a == red.a());
      um2::gmsh::model::getColor(2, 2, r, g, b, a);
      ASSERT(r == red.r() && g == red.g() && b == red.b() && a == red.a());
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
      um2::Color const blue("blue");
      int r = 0;
      int g = 0;
      int b = 0;
      int a = 0;
      um2::gmsh::model::getColor(2, 3, r, g, b, a);
      ASSERT(r == blue.r() && g == blue.g() && b == blue.b() && a == blue.a());
    }
    um2::gmsh::finalize();
  }
}

TEST_CASE(groupPresFragment_3d3d)
{
  std::vector<um2::Material> const materials = {um2::Material("Fuel", "red"),
                                                um2::Material("Moderator", "blue")};
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
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() && a == um2::red.a());
      um2::gmsh::model::getColor(3, 2, r, g, b, a);
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() && a == um2::red.a());
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
      ASSERT(r == um2::blue.r() && g == um2::blue.g() && b == um2::blue.b() && a == um2::blue.a());
    }
    um2::gmsh::finalize();
  }
}

TEST_CASE(groupPresIntersect_2d2d)
{
  std::vector<um2::Material> const materials = {um2::Material("Fuel", "red"),
                                                um2::Material("Moderator", "blue")};
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
      ASSERT(r == um2::red.r() && g == um2::red.g() && b == um2::red.b() && a == um2::red.a());
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

TEST_SUITE(gmsh_model)
{
  TEST(addToPhysicalGroup)
  TEST(getMaterials);
  TEST(groupPresFragment_2d2d);
  TEST(groupPresFragment_3d3d);
  TEST(groupPresIntersect_2d2d);
}
#endif // UM2_USE_GMSH

auto
main() -> int
{
#if UM2_USE_GMSH
  um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Error);
  RUN_SUITE(gmsh_model);
#endif
  return 0;
}
