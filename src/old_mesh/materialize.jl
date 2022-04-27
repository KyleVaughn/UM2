"""
    getpoints(ids::SVector{N, U}, points::Vector{<:Point}) where {N, U <: Unsigned}

Return an `SVector` of the `N` points at indices `ids`.
"""
@generated function getpoints(ids::SVector{N, U}, points::Vector{<:Point}
                             ) where {N, U <: Unsigned} 
    # This is generated for speed, since it's called a LOT. 
    # If there's a faster, way please remove this abomination.
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[ids[$i]], "
    end 
    points_string *= ")" 
    return Meta.parse(points_string)
end

# Return a LineSegment from the point IDs in an edge
function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
    return LineSegment(getpoints(edge, points))
end

# Return a QuadraticSegment from the point IDs in an edge
function materialize_edge(edge::SVector{3}, points::Vector{<:Point})
    return QuadraticSegment(getpoints(edge, points))
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh)
    return materialize_edge.(edges(mesh), Ref(mesh.points))
end

# Return a Polygon from the point IDs in a face
function materialize_polygon(face::SVector{N}, points::Vector{<:Point}) where {N}
    return Polygon{N}(getpoints(face, points))
end

# Return a QuadraticPolygon from the point IDs in a face
function materialize_quadratic_polygon(face::SVector{N}, points::Vector{<:Point}) where {N}
    return QuadraticPolygon{N}(getpoints(face, points))
end

# Return a the materialized face for mesh.faces[id] 
function materialize_face(id, mesh::PolygonMesh)
    return materialize_polygon(mesh.faces[id], mesh.points)
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::PolygonMesh)
    return materialize_polygon.(mesh.faces, Ref(mesh.points))
end

# Return a the materialized face for mesh.faces[id] 
function materialize_faces(id, mesh::QuadraticPolygonMesh)
    return materialize_quadratic_polygon(mesh.faces[id], mesh.points)
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::QuadraticPolygonMesh)
    return materialize_quadratic_polygon.(mesh.faces, Ref(mesh.points))
end
