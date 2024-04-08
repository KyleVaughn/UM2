#pragma once

#include <um2/config.hpp>

//==============================================================================
// Mandatory includes
//==============================================================================
//#include <um2/geometry/point.hpp>
#include <um2/physics/material.hpp>
#include <um2/mpact/model.hpp>
#include <um2/common/string_to_lattice.hpp>

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
initialize();

void
finalize();

} // namespace um2
