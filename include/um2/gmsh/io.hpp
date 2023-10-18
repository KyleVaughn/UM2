#pragma once

#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>

#  include <string> // std::string

namespace um2::gmsh
{

// Extend the gmsh::write function so that it has the option to preserve
// physical groups and colors via extra_info = true
void
write(std::string const & filename, bool extra_info);

void
open(std::string const & filename, bool extra_info);

} // namespace um2::gmsh
#endif // UM2_USE_GMSH
