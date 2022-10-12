export PolygonMesh, TriMesh, QuadMesh

export name, num_faces, face, face_iterator, faces, edges, bounding_box, 
       centroid, face_areas, sort_morton_order!, find_face, find_face_morton_order,
       find_face_robust_morton, vtk_type

# POLYGON MESH
# -----------------------------------------------------------------------------
#
# A 2D polygon mesh.
#
# N = 3 is a triangle mesh
# N = 4 is a quad mesh
#

struct PolygonMesh{N} <: AbstractMesh

    # Name of the mesh.
    name::String

    # Vertex positions.
    vertices::Vector{Point2{UM_F}}

    # Face-vertex connectivity.
    fv_conn::Vector{UM_I}

    # Vertex-face connectivity offsets.
    vf_offsets::Vector{UM_I}

    # Vertex-face connectivity.
    vf_conn::Vector{UM_I}

end

# -- Type aliases --

const TriMesh = PolygonMesh{3}
const QuadMesh = PolygonMesh{4}

# -- Constructors --

function polygon_mesh_vf_conn(N::Int64, nverts::Int64, fv_conn::Vector{UM_I})
    # Vertex-face connectivity
    nfaces = length(fv_conn) ÷ N
    vf_conn_vert_counts = zeros(UM_I, nverts)
    for face_id in 1:nfaces
        for ivert in 1:N
            vert_id = fv_conn[N * (face_id - 1) + ivert]
            vf_conn_vert_counts[vert_id] += 1
        end
    end
    vf_offsets = Vector{UM_I}(undef, nverts + 1)
    vf_offsets[1] = 1
    vf_offsets[2:end] .= cumsum(vf_conn_vert_counts)
    vf_offsets[2:end] .+= 1

    vf_conn_vert_counts .-= 1
    vf_conn = Vector{UM_I}(undef, vf_offsets[end] - 1)
    for face_id in 1:nfaces
        for ivert in 1:N
            vert_id = fv_conn[N * (face_id - 1) + ivert]
            vf_conn[vf_offsets[vert_id] + vf_conn_vert_counts[vert_id]] = face_id
            vf_conn_vert_counts[vert_id] -= 1
        end
    end
    for vert_id in 1:nverts
        this_offset = vf_offsets[vert_id]
        next_offset = vf_offsets[vert_id + 1]
        sort!(view(vf_conn, this_offset:(next_offset - 1)))
    end
    return (vf_offsets, vf_conn)
end

function PolygonMesh{N}(file::MeshFile) where {N}
    # Error checking
    if file.format == ABAQUS_FORMAT
        # Use vtk numerical types
        if N === 3
            vtk_type = VTK_TRIANGLE
        elseif N === 4
            vtk_type = VTK_QUAD
        else
            error("Unsupported polygon mesh type")
        end

        if any(eltype -> eltype !== vtk_type, file.element_types)
            error("Not all elements are VTK type " * string(vtk_type))
        end
    elseif file.format == XDMF_FORMAT
        # Use xdmf numerical types
        if N === 3
            xdmf_type = XDMF_TRIANGLE
        elseif N === 4
            xdmf_type = XDMF_QUAD
        else
            error("Unsupported polygon mesh type")
        end

        if any(eltype -> eltype !== xdmf_type, file.element_types)
            error("Not all elements are XDMF type " * string(xdmf_type))
        end
    else
        error("Unsupported mesh file format")
    end

    nfaces = length(file.elements) ÷ N
    if nfaces * N != length(file.elements)
        error("Number of elements is not a multiple of " * string(N))
    end

    # Vertices
    nverts = length(file.nodes)
    vertices = Vector{Point2{UM_F}}(undef, nverts)
    for i in 1:nverts
        vertices[i] = Point2{UM_F}(file.nodes[i][1], file.nodes[i][2])
    end

    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, nverts, file.elements)

    return PolygonMesh{N}(
        file.name,
        vertices,
        file.elements,
        vf_offsets,
        vf_conn
    )
end

# -- Basic properties --

name(mesh::PolygonMesh) = mesh.name

num_faces(mesh::PolygonMesh{N}) where {N} = length(mesh.fv_conn) ÷ N

function fv_conn(i::Integer, mesh::PolygonMesh{N}) where {N}
    return ntuple(j -> mesh.fv_conn[N * (i - 1) + j], Val(N))
end

fv_conn_iterator(mesh::PolygonMesh) = (fv_conn(i, mesh) for i in 1:num_faces(mesh))

function face(i::Integer, mesh::PolygonMesh{N}) where {N}
    return Polygon{N}(ntuple(j -> mesh.vertices[mesh.fv_conn[N * (i - 1) + j]],
                             Val(N))
                     )
end

face_iterator(mesh::PolygonMesh) = (face(i, mesh) for i in 1:num_faces(mesh))

faces(mesh::PolygonMesh) = collect(face_iterator(mesh))

function edges(mesh::PolygonMesh{N}) where {N}
    unique_edges = NTuple{2, UM_I}[]
    nedges = 0
    for fv_conn in fv_conn_iterator(mesh)
        for ev_conn in polygon_ev_conn_iterator(fv_conn)
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
    lines = Vector{LineSegment{2, UM_F}}(undef, nedges)
    for iedge = 1:nedges
        lines[iedge] = LineSegment(
            mesh.vertices[unique_edges[iedge][1]],
            mesh.vertices[unique_edges[iedge][2]]
        )
    end
    return lines
end

# -- In --

function find_face(P::Point2, mesh::PolygonMesh)
    for (i, fv_conn) in enumerate(fv_conn_iterator(mesh))
        if all(vids -> isCCW(P, mesh.vertices[vids[1]], mesh.vertices[vids[2]]),
               polygon_ev_conn_iterator(fv_conn))
            return i
        end
    end
    return 0
end

# Assumes the mesh has been sorted in morton order
function find_face_morton_order(P::Point2{UM_F}, 
                                mesh::PolygonMesh{N}, 
                                scale_inv::UM_F) where {N}
    lo, hi = morton_z_neighbors(P, mesh.vertices, scale_inv)
    dlo = distance2(P, mesh.vertices[lo])
    dhi = distance2(P, mesh.vertices[hi])
    if dlo < dhi
        vmin = vid = lo
        dmin = dlo
    else
        vmin = vid = hi
        dmin = dhi
    end
    # Check the faces that share the vertex.
    # Compute the closest vertex on each of these faces in case this fails.
    # If it fails, move to the next vertex and repeat.
    for _ = 1:4 # Try the initial vertex and the next closest vertex
        # For each face id in the vertex-face connectivity
        for voffset in mesh.vf_offsets[vid]:(mesh.vf_offsets[vid + 1] - 1)
            face_id = mesh.vf_conn[voffset]
            # Get the vertices that compose the face
            this_fv_conn = fv_conn(face_id, mesh)
            # Check if the point is inside the face
            if all(vids -> isCCW(P, mesh.vertices[vids[1]], mesh.vertices[vids[2]]),
                   polygon_ev_conn_iterator(this_fv_conn))
                return face_id
            end
            # If the point is not inside the face, check if any of the vertices
            # are closer than the current closest vertex
            for iv in 1:N
                d = distance2(P, mesh.vertices[this_fv_conn[iv]])
                if d < dmin
                    vmin = Int64(this_fv_conn[iv])
                    dmin = d
                end
            end
        end
        # If we get here, the point is not inside any of the faces that share
        # the vertex. Move to the next closest vertex and repeat.
        # If we have already checked the next closest vertex, we are out of
        # luck.
        if vmin === vid
            break
        end
        vid = vmin
    end
    # If we didn't find a face, return 0
    return 0
end

function find_face_robust_morton(P::Point2{UM_F}, 
                                 mesh::PolygonMesh{N}, 
                                 scale_inv::UM_F) where {N}
    face_id = find_face_morton_order(P, mesh, scale_inv)
    if face_id == 0
        face_id = find_face(P, mesh)
    end
    return face_id
end

# -- Bounding box --

bounding_box(mesh::PolygonMesh) = bounding_box(mesh.vertices)

# -- Centroid --

centroid(i::Integer, mesh::PolygonMesh) = centroid(face(i, mesh))

centroid_iterator(mesh::PolygonMesh) = (centroid(i, mesh) for i in 1:num_faces(mesh))

# -- Areas --

face_areas(mesh::PolygonMesh) = map(area, face_iterator(mesh))

# -- Morton ordering --

function sort_morton_order!(mesh::PolygonMesh{N}) where {N}
    bb = bounding_box(mesh)
    scale = max(width(bb), height(bb))
    scale_inv = 1 / scale
    nfaces = num_faces(mesh)

    # Sort the vertices
    point_map = sortperm_morton_order(mesh.vertices, scale_inv)
    point_map_inv = invperm(point_map)
    permute!(mesh.vertices, point_map)

    # Remap the face-vertex connectivity
    for i in 1:(N * nfaces)
        mesh.fv_conn[i] = point_map_inv[mesh.fv_conn[i]]
    end

    # Sort the faces by centroid
    centroids = collect(centroid_iterator(mesh))
    centroid_map = sortperm_morton_order(centroids, scale_inv)
    fv_map = Vector{eltype(centroid_map)}(undef, N * nfaces)
    for iface in 1:nfaces
        for ivert in 1:N
            fv_map[N * (iface - 1) + ivert] = mesh.fv_conn[N * (centroid_map[iface] - 1) + ivert]
        end
    end

    # Remap the face-vertex connectivity
    for i in 1:(N * nfaces)
        mesh.fv_conn[i] = fv_map[i]
    end

    # Recompute the vertex-face connectivity
    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, length(mesh.vertices), mesh.fv_conn)
    mesh.vf_offsets .= vf_offsets
    mesh.vf_conn .= vf_conn

    return nothing
end

function vtk_type(mesh::PolygonMesh{N}) where {N}
    if N === 3
        return VTK_TRIANGLE
    elseif N === 4
        return VTK_QUAD
    else
        error("Unsupported polygon type")
    end
end

# -- Show --

function Base.show(io::IO, mesh::PolygonMesh{N}) where {N}
    if N === 3
        poly_type = "Tri"
    elseif N === 4
        poly_type = "Quad"
    else
        poly_type = "Polygon"
    end 
    println(io, poly_type, "Mesh{", UM_F, ", ", UM_I, "}")
    println(io, "  ├─ Name      : ", mesh.name)    
    size_B = Base.summarysize(mesh)
    if size_B < 1e6    
        println(io, "  ├─ Size (KB) : ", string(@sprintf("%.3f", size_B/1000)))    
    else    
        println(io, "  ├─ Size (MB) : ", string(@sprintf("%.3f", size_B/1e6)))    
    end
    println(io, "  ├─ Vertices  : ", length(mesh.vertices))
    println(io, "  └─ Faces     : ", num_faces(mesh))
end
