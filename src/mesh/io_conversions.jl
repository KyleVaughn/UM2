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
    elseif file.format == XDMF_FORMAT
        if all(eltype -> eltype === XDMF_TRIANGLE, file.element_types)
            return PolygonMesh{3}(file)
        elseif all(eltype -> eltype === XDMF_QUAD, file.element_types)
            return PolygonMesh{4}(file)
        elseif all(eltype -> eltype === XDMF_QUADRATIC_TRIANGLE, file.element_types)
            return QuadraticPolygonMesh{6}(file)
        elseif all(eltype -> eltype === XDMF_QUADRATIC_QUAD, file.element_types)
            return QuadraticPolygonMesh{8}(file)
        else
            error("Unsupported element type")
        end
    else
        error("Unsupported format")
    end
end

function MeshFile(format::Int64,
                  mesh::PolygonMesh{N},
                  elsets::Dict{String, Set{UM_I}}) where {N}
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
    element_offsets = collect(UM_I(1):UM_I(N):UM_I(nfaces*N))
    elements = mesh.fv_conn
    return MeshFile(filepath, 
                    format, 
                    name, 
                    nodes, 
                    element_types,
                    element_offsets, 
                    elements,
                    elsets)
end

function MeshFile(format::Int64,
                  mesh::QPolygonMesh{N},
                  elsets::Dict{String, Set{UM_I}}) where {N}
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
    element_offsets = collect(UM_I(1):UM_I(N):UM_I(nfaces*N))
    elements = mesh.fv_conn
    return MeshFile(filepath, 
                    format, 
                    name, 
                    nodes, 
                    element_types,
                    element_offsets, 
                    elements,
                    elsets)
end

function HierarchicalMeshFile(format::Int64,
                              mesh::HierarchicalMesh{M},
                              leaf_elsets::Vector{Dict{String, Set{UM_I}}}) where {M}
    if format != XDMF_FORMAT
        error("Unsupported format")
    end

    filepath = ""
    partition_tree = mesh.partition_tree
    nleaves = length(mesh.leaf_meshes) 
    leaf_meshes = Vector{MeshFile}(undef, nleaves)
    for i in 1:nleaves
        leaf_meshes[i] = MeshFile(format, mesh.leaf_meshes[i], leaf_elsets[i])
    end
    return HierarchicalMeshFile(filepath,
                                format, 
                                partition_tree, 
                                leaf_meshes)
end

# HM from HMF
# Not type-stable, since M could be a PolygonMesh{N} or QPolygonMesh{N}
function HierarchicalMesh(hmf::HierarchicalMeshFile)
    leaf_meshes = map(x -> to_mesh(x), hmf.leaf_meshes)
    mesh_eltype = eltype(leaf_meshes)
    @assert mesh_eltype <: PolygonMesh || mesh_eltype <: QPolygonMesh
    return HierarchicalMesh{mesh_eltype}(hmf.partition_tree, leaf_meshes)
end

to_mesh(hmf::HierarchicalMeshFile) = HierarchicalMesh(hmf)
