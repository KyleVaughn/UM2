module Gmsh 

using Pkg.Artifacts

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
