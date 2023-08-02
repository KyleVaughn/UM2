#pragma once

#include <um2/config.hpp> // UM2_ENABLE_GMSH

#if UM2_ENABLE_GMSH
#  include <um2/common/Log.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/mesh/MeshType.hpp>

#  include <concepts> // std::floating_point, std::unsigned_integral
#  include <string>   // std::string
#  include <vector>   // std::vector

namespace um2::gmsh::model::mesh
{

void
setGlobalMeshSize(double size);

// void set_mesh_field_from_groups(
//         int const dim,
//         std::vector<std::string> const & groups,
//         std::vector<double> const & values);

void
generateMesh(MeshType mesh_type, int opt_iters = 5, int smooth_iters = 100);

} // namespace um2::gmsh::model::mesh
#endif // UM2_ENABLE_GMSH
