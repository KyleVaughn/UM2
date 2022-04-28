module Mesh

using LinearAlgebra
using StaticArrays
using ..Geometry
import ..Geometry: xmin, ymin, zmin, xmax, ymax, zmax

include("rectilinear_grid.jl")
##include("polytopemesh.jl")
end
