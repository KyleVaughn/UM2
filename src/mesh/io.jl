export import_mesh #, export_mesh

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

#
#function export_mesh(path::String, mesh)
#    @info "Writing " * path
#    if endswith(path, ".xdmf")
#        return write_xdmf(path, mesh)
#    else
#        error("Could not determine mesh file type from extension")
#    end
#end
