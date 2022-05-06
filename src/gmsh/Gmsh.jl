module Gmsh 

using Pkg.Artifacts

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

export gmsh

end
