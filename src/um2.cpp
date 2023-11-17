#include <um2.hpp>
#include <um2/common/log.hpp>

#include <algorithm>

namespace um2
{

void
#if UM2_USE_GMSH
initialize(String const & verbosity, bool init_gmsh, int gmsh_verbosity)
#else
initialize(String const & verbosity)
#endif
{
  Log::reset();
  // Set verbosity
  // Make uppercase for comparison
  String verbosity_upper = verbosity;
  std::transform(verbosity.data(), verbosity.data() + verbosity.size(),
                 verbosity_upper.data(), ::toupper);
  if (verbosity_upper == "TRACE") {
    Log::setMaxVerbosityLevel(LogVerbosity::Trace);
  } else if (verbosity_upper == "DEBUG") {
    Log::setMaxVerbosityLevel(LogVerbosity::Debug);
  } else if (verbosity_upper == "INFO") {
    Log::setMaxVerbosityLevel(LogVerbosity::Info);
  } else if (verbosity_upper == "WARN") {
    Log::setMaxVerbosityLevel(LogVerbosity::Warn);
  } else if (verbosity_upper == "ERROR") {
    Log::setMaxVerbosityLevel(LogVerbosity::Error);
  } else if (verbosity_upper == "OFF") {
    Log::setMaxVerbosityLevel(LogVerbosity::Off);
  } else {
    Log::setMaxVerbosityLevel(LogVerbosity::Info);
    Log::warn("Invalid verbosity level: " + verbosity + ". Defaulting to INFO.");
  }
  Log::info("Initializing UM2");
#if UM2_USE_GMSH
  if (init_gmsh && gmsh::isInitialized() == 0) {
    gmsh::initialize();
    gmsh::option::setNumber("General.NumThreads", 0);             // System default
    gmsh::option::setNumber("Geometry.OCCParallel", 1);           // Parallelize OCC
    gmsh::option::setNumber("General.Verbosity", gmsh_verbosity); // Errors + warnings
  }
#endif
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
}

void
finalize()
{
  Log::info("Finalizing UM2");
  Log::flush();
#if UM2_USE_GMSH
  if (gmsh::isInitialized() != 0) {
    gmsh::finalize();
  }
#endif
}

} // namespace um2
