#include <um2.hpp>
#include <um2/common/Log.hpp>

#include <algorithm>

namespace um2
{

void
#if UM2_USE_GMSH
initialize(std::string const & verbosity, bool init_gmsh, Int gmsh_verbosity)
#else
initialize(std::string const & verbosity)
#endif
{
  Log::reset();
  // Set verbosity
  // Make uppercase for comparison
  std::string verbosity_upper = verbosity;
  std::transform(verbosity.begin(), verbosity.end(), verbosity_upper.begin(), ::toupper);
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
    Log::warn("Invalid verbosity level: " + String(verbosity.c_str()) +
              ". Defaulting to INFO.");
  }
  Log::info("Initializing UM2");
#if UM2_USE_GMSH
  if (init_gmsh && gmsh::isInitialized() == 0) {
    gmsh::initialize();
    gmsh::option::setNumber("General.NumThreads", 0);   // System default
    gmsh::option::setNumber("Geometry.OCCParallel", 1); // Parallelize OCC
#  if UM2_ENABLE_INT64 == 1
    gmsh::option::setNumber("General.Verbosity",
                            static_cast<int>(gmsh_verbosity)); // Errors + warnings
#  else
    gmsh::option::setNumber("General.Verbosity", gmsh_verbosity); // Errors + warnings
#  endif
  }
#endif
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
