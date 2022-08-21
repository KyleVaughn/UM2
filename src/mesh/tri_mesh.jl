export TriMesh

# TRIMESH    
# -----------------------------------------------------------------------------    
#    
# A 2D triangle mesh.
#

struct TriMesh{T <: AbstractFloat, I <: Integer}

    # Name of the mesh.
    name::String

    # Vertex positions.
    vertices::Vector{Point{2, T}}

    # Face-vertex connectivity.
    fv_conn::Vector{I}

    # Vertex-face connectivity offsets.
    vf_offsets::Vector{I}

    # Vertex-face connectivity.
    vf_conn::Vector{I}

    # Material ID for each face.
    material_ids::Vector{Int8}

end

# -- Constructors --

function TriMesh(file::AbaqusFile{T, I}) where {T, I}
    # Error checking
    if any(eltype -> eltype !== VTK_TRIANGLE, file.element_types)
        error("Not all elements are triangles")
    end
    nfaces = length(file.elements) ÷ 3
    if nfaces * 3 != length(file.elements)
        error("Number of elements is not a multiple of 3")
    end


    # Vertices
    nverts = length(file.nodes)
    vertices = Vector{Point{2, T}}(undef, nverts)
    for i in 1:nverts
        vertices[i] = Point{2, T}(file.nodes[i][1], file.nodes[i][2])
    end

    # Vertex-face connectivity
    vf_offsets = Vector{I}(undef, nverts + 1)
    vf_conn_vecs = [ I[] for _ in 1:nverts ]
    for face_id in 1:nfaces
        for ivert in 1:3
            vert_id = file.elements[3 * (face_id - 1) + ivert]
            push!(vf_conn_vecs[vert_id], face_id)
        end
    end
    vf_conn_size = 1
    for vert_id in 1:nverts
        vf_offsets[vert_id] = vf_conn_size
        vf_conn_size += length(vf_conn_vecs[vert_id])
    end
    vf_offsets[nverts + 1] = vf_conn_size
    vf_conn = Vector{I}(undef, vf_conn_size - 1)
    ctr = 1
    for vert_id in 1:nverts
        sort!(vf_conn_vecs[vert_id])
        for face_id in vf_conn_vecs[vert_id]
            vf_conn[ctr] = face_id
            ctr += 1
        end
    end
    
    # Materials
    material_names = get_material_names(file)
    material_ids = fill(Int8(-1), nfaces)
    for mat_id in 1:length(material_names)
        for face in file.elsets[material_names[mat_id]]
            if material_ids[face] == -1
                material_ids[face] = mat_id
            else
                error("Face " * string(face) * " is both material" *
                      string(material_ids[face]) * " and " *
                      string(mat_id))
            end
        end
    end

    return TriMesh{T, I}(
        file.name,
        vertices,
        file.elements,
        vf_offsets,
        vf_conn,
        material_ids
    )
end
