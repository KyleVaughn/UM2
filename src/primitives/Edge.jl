abstract type Edge{Dim,Ord,T} end
const Edge2D = Edge{2}
const Edge3D = Edge{3}
Base.broadcastable(e::Edge) = Ref(e)
@generated function edgepoints(edge::SVector{N}, points::Vector{<:Point}) where {N}
    points_string = "SVector("
    for i ∈ 1:N
        points_string *= "points[edge[$i]], "
    end
    points_string *= ")"
    return Meta.parse(points_string)
end
