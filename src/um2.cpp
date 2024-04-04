#include <um2.hpp>
#include <um2/common/logger.hpp>

namespace um2
{

void
initialize()
{
  logger::reset();
  logger::info("Initializing UM2");
#if UM2_USE_GMSH
  if (gmsh::isInitialized() == 0) {
    gmsh::initialize();
    gmsh::option::setNumber("General.NumThreads",
                            0); // System default (i.e. OMP_NUM_THREADS)
    gmsh::option::setNumber("Geometry.OCCParallel", 1); // Parallelize OCC
    //gmsh::option::setNumber("General.Verbosity", 2);    // Errors + warnings
  }
#endif
}

void
finalize()
{
  logger::info("Finalizing UM2");
#if UM2_USE_GMSH
  if (gmsh::isInitialized() != 0) {
    gmsh::finalize();
  }
#endif
}

} // namespace um2
