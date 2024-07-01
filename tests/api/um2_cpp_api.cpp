#include <um2.hpp>
#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#endif

#include "../test_macros.hpp"

TEST_CASE(initialize_finalize)
{
  um2::initialize();
#if UM2_USE_GMSH
  ASSERT(um2::gmsh::isInitialized());
#endif
  um2::finalize();
}

TEST_SUITE(cpp_api) { TEST(initialize_finalize); }

auto
main() -> int
{
  RUN_SUITE(cpp_api);
  return 0;
}
