#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#endif

#include "../test_macros.hpp"

#if UM2_USE_GMSH

TEST_CASE(base_gmsh_api)
{
  um2::gmsh::initialize();
  int const tag = um2::gmsh::model::occ::addDisk(0, 0, 0, 1, 1);
  ASSERT(tag == 1);
  um2::gmsh::finalize();
}

TEST_SUITE(gmsh_wrapper) { TEST(base_gmsh_api); }
#endif // UM2_USE_GMSH

auto
main() -> int
{
#if UM2_USE_GMSH
  RUN_SUITE(gmsh_wrapper);
#endif
  return 0;
}
