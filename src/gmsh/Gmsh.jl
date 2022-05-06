module Gmsh 

using Pkg.Artifacts

gmsh_dir = readdir(artifact"gmsh", join=true)[1]
gmsh_jl = joinpath(gmsh_dir, "lib", "gmsh.jl") 
include(gmsh_jl)
export gmsh

include("model/safe_add_physical_group.jl")
include("model/get_entities_by_color.jl")

end
