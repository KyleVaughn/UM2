#pragma once

#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <um2/common/log.hpp>
#  include <um2/geometry/point.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/mpact/spatial_partition.hpp>
#  include <um2/physics/material.hpp>

namespace um2::gmsh::model
{

void
addToPhysicalGroup(int dim, std::vector<int> const & tags, int tag = -1,
                   std::string const & name = "");

void
getMaterials(std::vector<Material<Float>> & materials);

namespace occ
{

// A gmsh::model::occ::fragment that preserves the model's physical groups
//
// In the event that two overlapping entities have material physical groups, the
// optional material hierarchy is used to choose a single material for the resultant
// overlapping entity/entities.
void
groupPreservingFragment(gmsh::vectorpair const & object_dimtags,
                        gmsh::vectorpair const & tool_dimtags,
                        gmsh::vectorpair & out_dimtags,
                        std::vector<gmsh::vectorpair> & out_dimtags_map,
                        std::vector<Material<Float>> const & material_hierarchy = {},
                        int tag = -1, bool remove_object = true, bool remove_tool = true);

// A gmsh::model::occ::intersect that preserves the model's physical groups
//
// In the event that two overlapping entities have material physical groups, the
// optional material hierarchy is used to choose a single material for the resultant
// overlapping entity/entities.
void
groupPreservingIntersect(gmsh::vectorpair const & object_dimtags,
                         gmsh::vectorpair const & tool_dimtags,
                         gmsh::vectorpair & out_dimtags,
                         std::vector<gmsh::vectorpair> & out_dimtags_map,
                         std::vector<Material<Float>> const & material_hierarchy = {},
                         int tag = -1, bool remove_object = true,
                         bool remove_tool = true);

auto
addCylindricalPin2D(Point2d const & center, std::vector<double> const & radii,
                    std::vector<Material<Float>> const & materials) -> std::vector<int>;

auto
addCylindricalPinLattice2D(std::vector<std::vector<double>> const & radii,
                           std::vector<std::vector<Material<Float>>> const & materials,
                           std::vector<Vec2d> const & dxdy,
                           std::vector<std::vector<int>> const & pin_ids,
                           Point2d const & offset = {0.0, 0.0}) -> std::vector<int>;

auto
addCylindricalPin(Point3d const & center, double height,
                  std::vector<double> const & radii,
                  std::vector<Material<Float>> const & materials) -> std::vector<int>;

auto
addCylindricalPinLattice(std::vector<std::vector<double>> const & radii,
                         std::vector<std::vector<Material<Float>>> const & materials,
                         double height, std::vector<Vec2d> const & dxdy,
                         std::vector<std::vector<int>> const & pin_ids,
                         Point3d const & offset = {0.0, 0.0, 0.0}) -> std::vector<int>;

void
overlaySpatialPartition(mpact::SpatialPartition<Float, Int> const & partition,
                        std::string const & fill_material_name = "Moderator",
                        Color fill_material_color = Color("royalblue"));
} // namespace occ
} // namespace um2::gmsh::model
#endif // UM2_USE_GMSH
