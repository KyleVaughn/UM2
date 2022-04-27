module MOCNeutronTransport

#const minimum_ray_segment_length = 1e-4 # 1μm
#const visualization_enabled = true
#const visualize_ray_tracing = false 
#const plot_nonlinear_subdivisions = 3
#
#using CUDA, Colors, FixedPointNumbers, HDF5, Logging, LightXML, LinearAlgebra, 
#      StaticArrays, Statistics
using Reexport
using Pkg.Artifacts
#using Dates: now, format
#using LoggingExtras: TransformerLogger, global_logger
#
#import Base: +, -, *, /, ==, ≈, convert, hypot, intersect, issubset, sort, 
#             sort!, zero
#import LinearAlgebra: ×, ⋅, norm, inv

@reexport using LinearAlgebra
@reexport using StaticArrays

# include gmsh
# Check if there is a local install on JULIA_LOAD_PATH
for path in Base.load_path()
    gmsh_jl = joinpath(path, "gmsh.jl")
    if isfile(gmsh_jl)
        @info "MOCNeutronTransport is using the gmsh API found at: "*gmsh_jl
        include(gmsh_jl)
        break
    end
end
# Fallback on the SDK
if !@isdefined(gmsh)
    gmsh_dir = readdir(artifact"gmsh", join=true)[1]
    gmsh_jl = joinpath(gmsh_dir, "lib", "gmsh.jl") 
    if isfile(gmsh_jl)
        @info "MOCNeutronTransport is using the gmsh API found at: "*gmsh_jl
        include(gmsh_jl)
    else
        error("Could not find gmsh API.")
    end
end

include("quadrature/Quadrature.jl")
@reexport using .Quadrature
include("geometry/Geometry.jl")
@reexport using .Geometry
include("mesh/Mesh.jl")
@reexport using .Mesh
include("plot/Plot.jl")
@reexport using .Plot
end
