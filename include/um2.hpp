#pragma once

// ---------------------------------------------------------------------------
// Mandatory includes
// ---------------------------------------------------------------------------
#include <um2/config.hpp>

#include <string>

// ---------------------------------------------------------------------------
// Optional includes
// ---------------------------------------------------------------------------
#if UM2_ENABLE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/io.hpp>
#  include <um2/gmsh/model.hpp>
// #   include <um2/gmsh/mesh.hpp>
#endif

namespace um2
{

void
#if UM2_ENABLE_GMSH
initialize(std::string const & verbosity = "info", bool init_gmsh = true,
           int gmsh_verbosity = 2);
#else
initialize(std::string const & verbosity = "info");
#endif

void
finalize();

} // namespace um2
