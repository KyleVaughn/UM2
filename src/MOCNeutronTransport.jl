module MOCNeutronTransport

using EzXML
using HDF5
using Pkg.Artifacts
using Printf
using Reexport

using Colors: RGBA, N0f8
export RGBA, N0f8
using Dates: now, format
using LoggingExtras: TransformerLogger, global_logger

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
#include("MPACT/MPACT.jl")
#include("plot/Plot.jl")
#include("gmsh/Gmsh.jl")
include("raytracing/Raytracing.jl")
end
