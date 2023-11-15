#pragma once

//==============================================================================
// Mandatory includes
//==============================================================================
#include <um2/common/to_vecvec.hpp>
#include <um2/mpact/spatial_partition.hpp>

//==============================================================================
// Optional includes
//==============================================================================
#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/io.hpp>
#  include <um2/gmsh/mesh.hpp>
#  include <um2/gmsh/model.hpp>
#endif

namespace um2
{

void
#if UM2_USE_GMSH
initialize(String const & verbosity = "info", bool init_gmsh = true,
           int gmsh_verbosity = 2);
#else
initialize(String const & verbosity = "info");
#endif

void
finalize();

} // namespace um2
