abstract type Face{Dim,Ord,T} end
const Face2D = Face{2}
const Face3D = Face{3}
Base.broadcastable(f::Face) = Ref(f)
@generated function facepoints(face::SVector{N}, points::Vector{<:Point}) where {N}
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[face[$i]], "
    end
    points_string *= ")"
    return Meta.parse(points_string)
end
