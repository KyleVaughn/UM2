module Mesh

using LinearAlgebra
using StaticArrays
using ..Geometry

import Base: issubset
import ..Geometry: xmin, ymin, zmin, xmax, ymax, zmax

include("rectilinear_grid.jl")
include("polytope_vertex_mesh.jl")
include("materialize.jl")
include("io_abaqus.jl")
include("mesh_io.jl")
end
