#include <um2/common/logger.hpp>

#if UM2_USE_GMSH
#  include <um2/gmsh/io.hpp>
#endif

#include "../test_macros.hpp"

#include <fstream>

#if UM2_USE_GMSH

TEST_CASE(write_open)
{
  um2::gmsh::initialize();
  um2::gmsh::model::occ::addDisk(0.0, 0.0, 0.0, 1.0, 1.0);
  um2::gmsh::model::occ::synchronize();
  um2::gmsh::model::addPhysicalGroup(2, {1}, -1, "A");
  um2::gmsh::model::setColor(
      {
          {2, 1}
  },
      255, 0, 0, 255, /*recursive=*/true);
  um2::gmsh::write("test.brep", /*extra_info=*/true);
  um2::gmsh::finalize();
  um2::gmsh::initialize();
  um2::gmsh::open("test.brep", /*extra_info=*/true);
  um2::gmsh::vectorpair dimtags;
  um2::gmsh::model::getEntities(dimtags);
  ASSERT(dimtags.size() == 3);
  ASSERT(dimtags[0].first == 0);
  ASSERT(dimtags[0].second == 1);
  ASSERT(dimtags[1].first == 1);
  ASSERT(dimtags[1].second == 1);
  ASSERT(dimtags[2].first == 2);
  ASSERT(dimtags[2].second == 1);
  um2::gmsh::vectorpair physgroups;
  um2::gmsh::model::getPhysicalGroups(physgroups);
  ASSERT(physgroups.size() == 1);
  ASSERT(physgroups[0].first == 2);
  ASSERT(physgroups[0].second == 1);
  int r = 0;
  int g = 0;
  int b = 0;
  int a = 0;
  um2::gmsh::model::getColor(2, 1, r, g, b, a);
  ASSERT(r == 255);
  ASSERT(g == 0);
  ASSERT(b == 0);
  ASSERT(a == 255);
  r = 0;
  g = 0;
  b = 0;
  a = 0;
  um2::gmsh::model::getColor(1, 1, r, g, b, a);
  ASSERT(r == 255);
  ASSERT(g == 0);
  ASSERT(b == 0);
  ASSERT(a == 255);
  um2::gmsh::finalize();

  int const stat = std::remove("test.brep");
  ASSERT(stat == 0);
}

TEST_SUITE(gmsh_io) { TEST(write_open); }
#endif // UM2_USE_GMSH

auto
main() -> int
{
#if UM2_USE_GMSH
  um2::logger::level = um2::logger::levels::error;
  RUN_SUITE(gmsh_io);
#endif
  return 0;
}
