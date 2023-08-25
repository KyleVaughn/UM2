#include <um2.hpp>

#include "../test_macros.hpp"

TEST_CASE(initialize_finalize)
{
  um2::initialize();
#if UM2_USE_GMSH
  ASSERT(um2::gmsh::isInitialized());
  ASSERT(um2::Log::getMaxVerbosityLevel() == um2::LogVerbosity::Info);
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
