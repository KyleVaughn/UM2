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
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (verbosity_upper == "TRACE") {
    Log::setLevel(LogLevel::Trace);
  } else if (verbosity_upper == "DEBUG") {
    Log::setLevel(LogLevel::Debug);
  } else if (verbosity_upper == "INFO") {
    Log::setLevel(LogLevel::Info);
  } else if (verbosity_upper == "WARN") {
    Log::setLevel(LogLevel::Warn);
  } else if (verbosity_upper == "ERROR") {
    Log::setLevel(LogLevel::Error);
  } else if (verbosity_upper == "OFF") {
    Log::setLevel(LogLevel::Off);
  } else {
    Log::setLevel(LogLevel::Info);
    Log::warn("Invalid log level: " + verbosity + ". Defaulting to Info.");
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
#if UM2_USE_GMSH
  if (gmsh::isInitialized() != 0) {
    gmsh::finalize();
  }
#endif
}

} // namespace um2
