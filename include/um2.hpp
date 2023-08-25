#pragma once

//==============================================================================
// Mandatory includes
//==============================================================================
#include <um2/common/to_vecvec.hpp>
#include <um2/config.hpp>
#include <um2/mpact/SpatialPartition.hpp>
#include <um2/mpact/io.hpp>
#include <um2/physics/Material.hpp>

#include <string>

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
initialize(std::string const & verbosity = "info", bool init_gmsh = true,
           Int gmsh_verbosity = 2);
#else
initialize(std::string const & verbosity = "info");
#endif

void
finalize();

} // namespace um2
