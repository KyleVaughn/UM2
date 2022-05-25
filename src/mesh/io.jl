export import_mesh, export_mesh
"""
    import_mesh(path::String)
    import_mesh(path::String, ::Type{T}=Float64) where {T<:AbstractFloat}

Import a mesh from file. The float type of the mesh may be specified with a second argument.
File type is inferred from the extension.
"""
function import_mesh(path::String, ::Type{T}) where {T<:AbstractFloat}
    @info "Reading "*path
    if endswith(path, ".inp")
        return read_abaqus(path, T)
    else
        error("Could not determine mesh file type from extension.")
    end
end

import_mesh(path::String) = import_mesh(path, Float64)

function export_mesh(path::String, mesh)
    @info "Writing "*path
    if endswith(path, ".xdmf")
        return write_xdmf(path, mesh)
    else
        error("Could not determine mesh file type from extension")
    end
end
