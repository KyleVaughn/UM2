#pragma once

#include <um2/config.hpp> // UM2_USE_GMSH

#if UM2_USE_GMSH
#  include <um2/common/logger.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/mesh/element_types.hpp>
#  include <um2/physics/material.hpp>

namespace um2::gmsh::model::mesh
{

void
setGlobalMeshSize(double size);

void
generateMesh(MeshType mesh_type, int smooth_iters = 100);

//auto
//setMeshFieldFromGroups(int dim, std::vector<std::string> const & groups,
//                       std::vector<double> const & sizes) -> int;
//
auto
setMeshFieldFromKnudsenNumber(
    int dim, um2::Vector<Material> const & materials, double kn_target,
    double mfp_threshold = -1.0, double mfp_scale = -1.0,
    std::vector<int> const & is_fuel = {}, // 1 for fuel, 0 for moderator
    XSecReduction strategy = XSecReduction::Mean) -> int;


} // namespace um2::gmsh::model::mesh
#endif // UM2_USE_GMSH
