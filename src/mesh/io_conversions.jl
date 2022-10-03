# to_mesh is not type-stable
function to_mesh(file::MeshFile)
    if file.format == ABAQUS_FORMAT
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
    else
        error("Unsupported format")
    end
end

function MeshFile(format::Int64,
                  mesh::PolygonMesh{N, T, I},
                  elsets::Dict{String, Set{I}}) where {N, T, I}
    nfaces = num_faces(mesh)
    if format == ABAQUS_FORMAT
        if N === 3
            element_types = fill(VTK_TRIANGLE, nfaces)
        elseif N === 4
            element_types = fill(VTK_QUAD, nfaces)
        else
            error("Unsupported element type")
        end
    elseif format == XDMF_FORMAT
        if N === 3
            element_types = fill(XDMF_TRIANGLE, nfaces)
        elseif N === 4
            element_types = fill(XDMF_QUAD, nfaces)
        else
            error("Unsupported element type")
        end
    else
        error("Unsupported format")
    end
    filepath = ""
    name = mesh.name
    nodes = mesh.vertices
    element_offsets = collect(I(1):I(N):I(nfaces*N))
    elements = mesh.fv_conn
    return MeshFile{T, I}(filepath, 
                          format, 
                          name, 
                          nodes, 
                          element_types,
                          element_offsets, 
                          elements,
                          elsets)
end

function MeshFile(format::Int64,
                  mesh::QPolygonMesh{N, T, I},
                  elsets::Dict{String, Set{I}}) where {N, T, I}
    nfaces = num_faces(mesh)
    if format == ABAQUS_FORMAT
        if N === 6
            element_types = fill(VTK_QUADRATIC_TRIANGLE, nfaces)
        elseif N === 8
            element_types = fill(VTK_QUADRATIC_QUAD, nfaces)
        else
            error("Unsupported element type")
        end
    elseif format == XDMF_FORMAT
        if N === 6
            element_types = fill(XDMF_QUADRATIC_TRIANGLE, nfaces)
        elseif N === 8
            element_types = fill(XDMF_QUADRATIC_QUAD, nfaces)
        else
            error("Unsupported element type")
        end
    else
        error("Unsupported format")
    end
    filepath = ""
    name = mesh.name
    nodes = mesh.vertices
    element_offsets = collect(I(1):I(N):I(nfaces*N))
    elements = mesh.fv_conn
    return MeshFile{T, I}(filepath, 
                          format, 
                          name, 
                          nodes, 
                          element_types,
                          element_offsets, 
                          elements,
                          elsets)
end
