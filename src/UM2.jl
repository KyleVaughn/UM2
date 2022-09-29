module UM2

using Logging
using Pkg.Artifacts
using Printf

using Dates: now, format
using LoggingExtras: TransformerLogger

# Gmsh
gmsh_dir = readdir(artifact"gmsh", join = true)[1]
gmsh_jl = joinpath(gmsh_dir, "lib", "gmsh.jl")
include(gmsh_jl)
export gmsh

include("common/defines.jl")
include("common/typedefs.jl")
include("common/constants.jl")
include("common/instructions.jl")
include("common/log.jl")
include("common/colors.jl")
#include("common/tree.jl")

include("math/vec.jl")
include("math/mat.jl")
include("math/morton.jl")

include("geometry/point.jl")
include("geometry/axis_aligned_box.jl")
include("geometry/line_segment.jl")
include("geometry/quadratic_segment.jl")
include("geometry/polygon/polygon.jl")
include("geometry/polygon/triangle.jl")
include("geometry/polygon/quadrilateral.jl")
include("geometry/quadratic_polygon/quadratic_polygon.jl")
include("geometry/quadratic_polygon/quadratic_triangle.jl")
include("geometry/quadratic_polygon/quadratic_quadrilateral.jl")
include("geometry/morton.jl")

include("mesh/abstract_mesh.jl")
include("mesh/cell_types.jl")
include("mesh/io_abaqus.jl")
include("mesh/rectilinear_grid.jl")
include("mesh/polygon_mesh.jl")
include("mesh/quadratic_polygon_mesh.jl")
include("mesh/submesh.jl")
include("mesh/io.jl")

include("ray_casting/ray.jl")
include("ray_casting/angular_quadrature.jl")
include("ray_casting/modular_rays.jl")
include("ray_casting/intersect/ray-line_segment.jl")
include("ray_casting/intersect/ray-quadratic_segment.jl")
include("ray_casting/intersect/ray-polygon_mesh.jl")
include("ray_casting/intersect/ray-quadratic_polygon_mesh.jl")

include("physics/material.jl")

include("mpact/grid_hierarchy.jl")
include("mpact/mesh_hierarchy.jl")

include("gmsh/model/get_entities_by_color.jl")
include("gmsh/model/color_material_physical_group_entities.jl")
include("gmsh/model/safe_add_physical_group.jl")
include("gmsh/model/add_cad_names_to_physical_groups.jl")
include("gmsh/model/add_materials_to_physical_groups_by_color.jl")
include("gmsh/model/import_model.jl")
include("gmsh/model/safe_fragment.jl")
include("gmsh/model/overlay_mpact_grid_hierarchy.jl")

include("gmsh/mesh/set_mesh_field_by_material.jl")
include("gmsh/mesh/generate_mesh.jl")
include("gmsh/mesh/get_cad_to_mesh_errors.jl")

end
