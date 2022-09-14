export QuadraticPolygonMesh, QPolygonMesh, QTriMesh, QQuadMesh

export num_faces, face, face_iterator, faces, edges, bounding_box

# QUADRATIC POLYGON MESH
# -----------------------------------------------------------------------------
#
# A 2D quadratic polygon mesh.
#
# N = 6 is a triangle mesh
# N = 8 is a quad mesh
#

struct QuadraticPolygonMesh{N, T <: AbstractFloat, I <: Integer}

    # Name of the mesh.
    name::String

    # Vertex positions.
    vertices::Vector{Point2{T}}

    # Face-vertex connectivity.
    fv_conn::Vector{I}

    # Vertex-face connectivity offsets.
    vf_offsets::Vector{I}

    # Vertex-face connectivity.
    vf_conn::Vector{I}

    # Material ID for each face.
    material_ids::Vector{Int8}

end

# -- Type aliases --

const QPolygonMesh = QuadraticPolygonMesh
const QTriMesh = QuadraticPolygonMesh{6}
const QQuadMesh = QuadraticPolygonMesh{8}

# -- Constructors --

function QuadraticPolygonMesh{N}(file::AbaqusFile{T, I}) where {N, T, I}
    # Error checking
    if N === 6
        vtk_type = VTK_QUADRATIC_TRIANGLE
    elseif N === 8
        vtk_type = VTK_QUADRATIC_QUAD
    else
        error("Unsupported quadratic polygon mesh type")
    end

    if any(eltype -> eltype !== vtk_type, file.element_types)
        error("Not all elements are VTK type " * string(vtk_type))
    end

    nfaces = length(file.elements) ÷ N
    if nfaces * N != length(file.elements)
        error("Number of elements is not a multiple of " * string(N))
    end

    # Vertices
    nverts = length(file.nodes)
    vertices = Vector{Point2{T}}(undef, nverts)
    for i in 1:nverts
        vertices[i] = Point2{T}(file.nodes[i][1], file.nodes[i][2])
    end

    # Vertex-face connectivity
    vf_conn_vert_counts = zeros(I, nverts)
    for face_id in 1:nfaces
        for ivert in 1:N
            vert_id = file.elements[N * (face_id - 1) + ivert]
            vf_conn_vert_counts[vert_id] += 1
        end
    end
    vf_offsets = Vector{I}(undef, nverts + 1)
    vf_offsets[1] = 1
    vf_offsets[2:end] .= cumsum(vf_conn_vert_counts)
    vf_offsets[2:end] .+= 1

    vf_conn_vert_counts .-= 1
    vf_conn = Vector{I}(undef, vf_offsets[end] - 1)
    for face_id in 1:nfaces
        for ivert in 1:N
            vert_id = file.elements[N * (face_id - 1) + ivert]
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

    return QuadraticPolygonMesh{N, T, I}(
        file.name,
        vertices,
        file.elements,
        vf_offsets,
        vf_conn,
        material_ids
    )
end

function QuadraticPolygonMesh{N}(file::String) where {N}
    abaqus_file = AbaqusFile(file)
    return (get_material_names(abaqus_file),
            QuadraticPolygonMesh{N}(abaqus_file))
end

# -- Basic properties --

num_faces(mesh::QPolygonMesh) = length(mesh.material_ids)

function fv_conn(i::Integer, mesh::QPolygonMesh{N}) where {N}
    return ntuple(j -> mesh.fv_conn[N * (i - 1) + j], Val(N))
end

fv_conn_iterator(mesh::QPolygonMesh) = (fv_conn(i, mesh) for i in 1:num_faces(mesh))

function face(i::Integer, mesh::QPolygonMesh{N}) where {N}
    return QPolygon{N}(ntuple(j -> mesh.vertices[mesh.fv_conn[N * (i - 1) + j]],
                             Val(N))
                     )
end

face_iterator(mesh::QPolygonMesh) = (face(i, mesh) for i in 1:num_faces(mesh))

faces(mesh::QPolygonMesh) = collect(face_iterator(mesh))

function edges(mesh::QPolygonMesh{N, T, I}) where {N, T, I}
    unique_edges = NTuple{3, I}[]
    nedges = 0
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in qpolygon_ev_conn_iterator(fv_conn)
            # Sort the edge vertices
            if ev_conn[1] < ev_conn[2]
                sorted_ev = ev_conn
            else
                sorted_ev = (ev_conn[2], ev_conn[1], ev_conn[3])
            end
            index = searchsortedfirst(unique_edges, sorted_ev)
            if nedges < index || unique_edges[index] !== sorted_ev
                insert!(unique_edges, index, sorted_ev)
                nedges += 1
            end
        end
    end
    lines = Vector{QuadraticSegment{2, T}}(undef, nedges)
    for iedge = 1:nedges
        lines[iedge] = QuadraticSegment(
            mesh.vertices[unique_edges[iedge][1]],
            mesh.vertices[unique_edges[iedge][2]],
            mesh.vertices[unique_edges[iedge][3]]
        )
    end
    return lines
end

# -- In --

function Base.in(P::Point2, mesh::QPolygonMesh)
    for (i, face) in enumerate(face_iterator(mesh))
        if P in face
            return i
        end
    end
    return 0
end

# -- Bounding box --

bounding_box(mesh::QPolygonMesh) = mapreduce(bounding_box, Base.union, face_iterator(mesh))

# -- Centroid --

centroid(i::Integer, mesh::QPolygonMesh) = centroid(face(i, mesh))
