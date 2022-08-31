export QuadMesh

export num_faces, faces, edges

# QUAD MESH
# -----------------------------------------------------------------------------
#
# A 2D quadrilateral mesh.
#

struct QuadMesh{T <: AbstractFloat, I <: Integer}

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

function QuadMesh(file::AbaqusFile{T, I}) where {T, I}
    # Error checking
    if any(eltype -> eltype !== VTK_QUAD, file.element_types)
        error("Not all elements are quadrilaterals")
    end
    nfaces = length(file.elements) ÷ 4
    if nfaces * 4 != length(file.elements)
        error("Number of elements is not a multiple of 4")
    end

    # Vertices
    nverts = length(file.nodes)
    vertices = Vector{Point{2, T}}(undef, nverts)
    for i in 1:nverts
        vertices[i] = Point{2, T}(file.nodes[i][1], file.nodes[i][2])
    end

    # Vertex-face connectivity
    vf_conn_vert_counts = zeros(I, nverts)
    for face_id in 1:nfaces
        for ivert in 1:4
            vert_id = file.elements[4 * (face_id - 1) + ivert]
            vf_conn_vert_counts[vert_id] += 1
        end
    end
    vf_offsets = Vector{I}(undef, nverts + 1)
    vf_offsets[1] = 1
    vf_offsets[2:end] = cumsum(vf_conn_vert_counts)
    vf_offsets[2:end] .+= 1

    vf_conn_vert_counts .-= 1
    vf_conn = Vector{I}(undef, vf_offsets[end] - 1)
    for face_id in 1:nfaces
        for ivert in 1:4
            vert_id = file.elements[4 * (face_id - 1) + ivert]
            vf_conn[vf_offsets[vert_id] + vf_conn_vert_counts[vert_id]] = face_id
            vf_conn_vert_counts[vert_id] -= 1
        end
    end
    for vert_id in 1:nverts
        this_offset = vf_offsets[vert_id]
        next_offset = vf_offsets[vert_id + 1]
        sort!(view(vf_conn, this_offset:(next_offset - 1)))
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

    return QuadMesh{T, I}(
        file.name,
        vertices,
        file.elements,
        vf_offsets,
        vf_conn,
        material_ids
    )
end

function QuadMesh(file::String)
    return QuadMesh(AbaqusFile(file))
end

# -- Iterators/Materialization --

num_faces(mesh::QuadMesh) = length(mesh.material_ids)

function fv_conn(i::Integer, mesh::QuadMesh)
    return ( 
        mesh.fv_conn[4 * i - 3],
        mesh.fv_conn[4 * i - 2],
        mesh.fv_conn[4 * i - 1],
        mesh.fv_conn[4 * i    ]
    )
end

fv_conn_iterator(mesh::QuadMesh) = (fv_conn(i, mesh) for i in 1:num_faces(mesh))

function face(i::Integer, mesh::QuadMesh)
    return Quadangle(
        mesh.vertices[mesh.fv_conn[4 * i - 3]],
        mesh.vertices[mesh.fv_conn[4 * i - 2]],
        mesh.vertices[mesh.fv_conn[4 * i - 1]],
        mesh.vertices[mesh.fv_conn[4 * i    ]]
    )
end

face_iterator(mesh::QuadMesh) = (face(i, mesh) for i in 1:num_faces(mesh))

faces(mesh::QuadMesh) = collect(face_iterator(mesh))

function edges(mesh::QuadMesh{T, I}) where {T, I}
    unique_edges = NTuple{2, I}[]
    nedges = 0
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in ev_conn_iterator(fv_conn)
            # Sort the edge vertices
            if ev_conn[1] < ev_conn[2]
                sorted_ev = ev_conn
            else
                sorted_ev = (ev_conn[2], ev_conn[1])
            end
            index = searchsortedfirst(unique_edges, sorted_ev)
            if nedges < index || unique_edges[index] !== sorted_ev
                insert!(unique_edges, index, sorted_ev)
                nedges += 1
            end
        end
    end
    lines = Vector{LineSegment{2, T}}(undef, nedges)
    for iedge = 1:nedges
        lines[iedge] = LineSegment(
            mesh.vertices[unique_edges[iedge][1]],
            mesh.vertices[unique_edges[iedge][2]]
        )
    end
    return lines
end

# -- In --

function Base.in(P::Point{2}, mesh::QuadMesh)
    for fv_conn in fv_conn_iterator(mesh)    
        if all(vids -> isCCW(P, mesh.vertices[vids[1]], mesh.vertices[vids[2]]), 
               ev_conn_iterator(fv_conn))
            return true
        end
    end
    return false
end
