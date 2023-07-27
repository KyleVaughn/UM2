#include <um2.hpp>
// #include <um2/common/log.hpp>

// NOLINTBEGIN
namespace um2
{

void
initialize(std::string const & verbosity, bool init_gmsh)
{
  int gmsh_verbosity = 0;
  if (verbosity == "info") {
    gmsh_verbosity = 2;
  }
  // Reset log
//    Log::reset();
// Set verbosity
// Make uppercase for comparison
//    thrust::transform(verbosity.begin(), verbosity.end(),
//                      verbosity.begin(), ::toupper);
//    if (verbosity == "DEBUG") {
//        Log::set_max_verbosity_level(LogVerbosity::debug);
//    } else if (verbosity == "INFO") {
//        Log::set_max_verbosity_level(LogVerbosity::info);
//    } else if (verbosity == "WARN") {
//        Log::set_max_verbosity_level(LogVerbosity::warn);
//    } else if (verbosity == "ERROR") {
//        Log::set_max_verbosity_level(LogVerbosity::error);
//    } else {
//        Log::set_max_verbosity_level(LogVerbosity::info);
//        Log::warn("Invalid verbosity level: " + verbosity + ". Defaulting to INFO.");
//    }
//    Log::info("Initializing UM2");
#if UM2_ENABLE_GMSH
  if (init_gmsh && gmsh::isInitialized() == 0) {
    gmsh::initialize();
    gmsh::option::setNumber("General.NumThreads", 0);             // System default
    gmsh::option::setNumber("Geometry.OCCParallel", 1);           // Parallelize OCC
    gmsh::option::setNumber("General.Verbosity", gmsh_verbosity); // Errors + warnings
  }
#endif
}

void
finalize()
{
//  Log::info("Finalizing UM2");
//  Log::flush();
#if UM2_ENABLE_GMSH
  if (gmsh::isInitialized() != 0) {
    gmsh::finalize();
  }
#endif
}
// NOLINTEND

} // namespace um2
