module UM2

include("common/defines.jl")
include("common/typedefs.jl")
include("common/constants.jl")
include("common/instructions.jl")

include("math/vec.jl")
include("math/mat.jl")
include("math/hilbert.jl")
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

include("mesh/cell_types.jl")
include("mesh/io_abaqus.jl")
include("mesh/polygon_mesh.jl")

#include("ray_casting/ray.jl")
#include("ray_casting/angular_quadrature.jl")
#include("ray_casting/modular_rays.jl")
#include("ray_casting/intersect/ray-line_segment.jl")
#include("ray_casting/intersect/ray-quadratic_segment.jl")
#include("ray_casting/intersect/ray-tri_mesh.jl")
#include("ray_casting/intersect/ray-quad_mesh.jl")
#
#include("physics/Physics.jl")
#include("mpact/MPACT.jl")
#include("gmsh/Gmsh.jl")
end
