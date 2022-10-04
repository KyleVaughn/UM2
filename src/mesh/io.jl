export import_mesh, export_mesh

# -- Import --

function import_mesh(path::String)
    @info "Reading " * path
    if endswith(path, ".inp")
        file = read_abaqus_file(path)
        return file.elsets, to_mesh(file)
#    elseif endswith(path, ".xdmf")
#        return read_xdmf(path, T)
    else
        error("Could not determine mesh file type from extension.")
    end
end

# -- Export --

function export_mesh(mesh::AbstractMesh, 
                     elsets::Dict{String, Set{I}},
                     path::String) where {I}
    @info "Writing " * path
    if endswith(path, ".xdmf")
        mf = MeshFile(XDMF_FORMAT, mesh, elsets)
        return write_xdmf(mf, path)
    else
        error("Could not determine mesh file type from extension")
    end
end
