module MOCNeutronTransport

#const minimum_ray_segment_length = 1e-4 # 1μm
#const visualize_ray_tracing = false 
#
#using CUDA, HDF5, Logging, LightXML,


using Pkg.Artifacts
using Printf
using Reexport

using Colors: RGBA
using FixedPointNumbers: N0f8
export RGBA, N0f8

#using Dates: now, format
#using LoggingExtras: TransformerLogger, global_logger
@reexport using LinearAlgebra
@reexport using StaticArrays
@reexport using Statistics

gmsh_dir = readdir(artifact"gmsh", join=true)[1]
gmsh_jl = joinpath(gmsh_dir, "lib", "gmsh.jl")
include(gmsh_jl)
export gmsh

include("common/Common.jl")
include("quadrature/Quadrature.jl")
include("geometry/Geometry.jl")
include("mesh/Mesh.jl")
include("plot/Plot.jl")
include("gmsh/Gmsh.jl")
end
