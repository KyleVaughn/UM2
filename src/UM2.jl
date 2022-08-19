module UM2

const UM2_ENABLE_VIS = false 

include("common/types.jl")

include("math/vec.jl")
include("math/mat.jl")

include("geometry/point.jl")
include("geometry/axis_aligned_box.jl")
include("geometry/abstract_types.jl")
include("geometry/line_segment.jl")
include("geometry/quadratic_segment.jl")
include("geometry/triangle.jl")
include("geometry/quadrilateral.jl")
include("geometry/quadratic_triangle.jl")
include("geometry/quadratic_quadrilateral.jl")

include("mesh/abstract_types.jl")
include("mesh/tri_mesh.jl")

if UM2_ENABLE_VIS
    include("vis/setup.jl")
    include("vis/point.jl")
    include("vis/abstract_edge.jl")
    include("vis/axis_aligned_box.jl")
    include("vis/abstract_face.jl")
end
#include("quadrature/Quadrature.jl")
#include("raytracing/Raytracing.jl")
#include("physics/Physics.jl")
#include("mpact/MPACT.jl")
#include("gmsh/Gmsh.jl")
end
