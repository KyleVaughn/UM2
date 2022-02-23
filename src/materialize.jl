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

# Return a LineSegment from the point IDs in an edge
function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
    return LineSegment(edgepoints(edge, points))
end

