export import_mesh #, export_mesh


# Not type-stable
function to_mesh(file::AbaqusFile)
    if all(eltype -> eltype === VTK_TRIANGLE, file.element_types)
        return PolygonMesh{3}(file) 
    elseif all(eltype -> eltype === VTK_QUAD, file.element_types)
        return PolygonMesh{4}(file)
    elseif all(eltype -> eltype === VTK_QUADRATIC_TRIANGLE, file.element_types)
        return QuadraticPolygonMesh{6}(file)
    elseif all(eltype -> eltype === VTK_QUADRATIC_QUAD, file.element_types)
        return QuadraticPolygonMesh{8}(file)
    else
        error("Unsupported element type")
    end
end
function import_mesh(path::String)
    @info "Reading " * path
    if endswith(path, ".inp")
        file = AbaqusFile(path)
        return file.elsets, to_mesh(file)
#    elseif endswith(path, ".xdmf")
#        return read_xdmf(path, T)
    else
        error("Could not determine mesh file type from extension.")
    end
end
#
#function export_mesh(path::String, mesh)
#    @info "Writing " * path
#    if endswith(path, ".xdmf")
#        return write_xdmf(path, mesh)
#    else
#        error("Could not determine mesh file type from extension")
#    end
#end
