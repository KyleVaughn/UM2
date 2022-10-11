export import_mesh, export_mesh

# -- Import --

# Not type-stable
function import_mesh(path::String)
    if endswith(path, ".inp")
        file = read_abaqus_file(path)
        return file.elsets, to_mesh(file)
    elseif endswith(path, ".xdmf")
        file = read_xdmf_file(path)
        if file isa MeshFile
            return file.elsets, to_mesh(xdmf_file)
        else # HierarchicalMeshFile
            hmf = to_mesh(file)
            leaf_elsets = [mf.elsets for mf in file.leaf_meshes]
            return leaf_elsets, hmf
        end
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
