#pragma once

#include <um2/config.hpp>

#if UM2_USE_GMSH
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/mpact/model.hpp>
#  include <um2/physics/material.hpp>

namespace um2::gmsh::model
{

void
addToPhysicalGroup(int dim, std::vector<int> const & tags, int tag = -1,
                   std::string const & name = "");

void
getMaterials(std::vector<Material> & materials);

namespace occ
{

// A gmsh::model::occ::fragment that preserves the model's D-dimensional physical       
// groups when fragmenting D-dimensional entities. All other physical groups are       
// destroyed.       
//       
// In the event that two overlapping entities have material physical groups, the       
// optional material hierarchy is used to choose a single material for the       
// resultant overlapping entity/entities.   
void
groupPreservingFragment(gmsh::vectorpair const & object_dimtags,
                        gmsh::vectorpair const & tool_dimtags,
                        gmsh::vectorpair & out_dimtags,
                        std::vector<gmsh::vectorpair> & out_dimtags_map,
                        std::vector<Material> const & material_hierarchy = {},
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
                         std::vector<Material> const & material_hierarchy = {},
                         int tag = -1, bool remove_object = true,
                         bool remove_tool = true);


//==============================================================================
// Functions for user-level API. These functions should take um2::Vector, um2::String,
// etc. Not std::vector, std::string, etc.
//==============================================================================

//auto
//addCylindricalPin2D(Vec2d const & center, std::vector<double> const & radii,
//                    std::vector<Material> const & materials) -> std::vector<int>;

auto
addCylindricalPin2D(Vec2F const & center, Vector<Float> const & radii,
                    Vector<Material> const & materials) -> um2::Vector<Int>;

auto
addCylindricalPinLattice2D(Vector<Vector<Int>> const & pin_ids,
                           Vector<Vec2F> const & xy_extents,
                           Vector<Vector<Float>> const & radii,
                           Vector<Vector<Material>> const & materials,
                           Vec2F const & offset = {0, 0}) -> Vector<Int>;

//auto
//addCylindricalPin(Vec3d const & center, double height, std::vector<double> const & radii,
//                  std::vector<Material> const & materials) -> std::vector<int>;
//
//auto
//addCylindricalPinLattice(std::vector<std::vector<double>> const & radii,
//                         std::vector<std::vector<Material>> const & materials,
//                         double height, std::vector<Vec2d> const & dxdy,
//                         std::vector<std::vector<int>> const & pin_ids,
//                         Vec3d const & offset = {0.0, 0.0, 0.0}) -> std::vector<int>;

void
overlayCoarseGrid(mpact::Model const & model, Material const & fill_material);
} // namespace occ
} // namespace um2::gmsh::model
#endif // UM2_USE_GMSH
