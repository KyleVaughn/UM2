#pragma once

#include <um2/config.hpp> // UM2_USE_GMSH

#if UM2_USE_GMSH
#  include <um2/common/Log.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/mesh/MeshFile.hpp>

#  include <concepts> // std::floating_point, std::unsigned_integral
#  include <string>   // std::string
#  include <vector>   // std::vector

namespace um2::gmsh::model::mesh
{

void
setGlobalMeshSize(double size);

auto
setMeshFieldFromGroups(int dim, std::vector<std::string> const & groups,
                       std::vector<double> const & sizes) -> std::vector<int>;

void
generateMesh(MeshType mesh_type, int smooth_iters = 100);
//generateMesh(MeshType mesh_type, int opt_iters = 5, int smooth_iters = 100);

} // namespace um2::gmsh::model::mesh
#endif // UM2_USE_GMSH
