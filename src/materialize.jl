@generated function edgepoints(edge::SVector{N}, points::Vector{<:Point}) where {N} 
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[edge[$i]], "
    end 
    points_string *= ")" 
    return Meta.parse(points_string)
end

@generated function facepoints(face::SVector{N}, points::Vector{<:Point}) where {N}
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[face[$i]], "
    end
    points_string *= ")"
    return Meta.parse(points_string)
end


# Return an SVector of the points in the face
function facepoints(face_id, mesh::UnstructuredMesh)
    return facepoints(mesh.faces[face_id], mesh.points)
end



# Return a LineSegment from the point IDs in an edge
function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
    return LineSegment(edgepoints(edge, points))
end

# Materialize edge 
# ---------------------------------------------------------------------------------------------
# Return a QuadraticSegment from the point IDs in an edge
function materialize_edge(edge::SVector{3}, points::Vector{<:Point})
    return QuadraticSegment(edgepoints(edge, points))
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh)
    return materialize_edge.(edges(mesh), Ref(mesh.points))
end

# Return a materialized facee for each facee in the mesh
function materialize_faces(mesh::UnstructuredMesh)
    return materialize_face.(1:length(mesh.faces), mesh)
end
