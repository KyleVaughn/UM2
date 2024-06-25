#include <um2.hpp>
#include <um2/common/logger.hpp>
#include <um2/config.hpp>

#include <cstdio>
#include <unistd.h>

#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#endif

namespace um2
{

void
initialize()
{
  logger::reset();
  // NOLINTNEXTLINE(misc-include-cleaner)
  if (isatty(fileno(stdout)) == 0 || isatty(fileno(stderr)) == 0) {
    logger::colorized = false;
  }
  logger::info("Initializing UM2");
#if UM2_USE_GMSH
  if (gmsh::isInitialized() == 0) {
    gmsh::initialize();
    gmsh::option::setNumber("General.NumThreads",
                            0); // System default (i.e. OMP_NUM_THREADS)
    gmsh::option::setNumber("Geometry.OCCParallel", 1); // Parallelize OCC
    // gmsh::option::setNumber("General.Verbosity", 2);    // Errors + warnings
  }
#endif
}

void
finalize()
{
  logger::info("Finalizing UM2");
  if (fflush(stdout) != 0) {
    logger::error("fflush(stdout) failed");
  }
#if UM2_USE_GMSH
  if (gmsh::isInitialized() != 0) {
    gmsh::finalize();
  }
#endif
}

} // namespace um2
