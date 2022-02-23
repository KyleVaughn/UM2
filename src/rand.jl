# Random
# ---------------------------------------------------------------------------------------------
# Random line in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}) where {Dim, F}
    points = rand(Point{Dim, F}, 2)
    return LineSegment{Dim, F}(points[1], points[2])
end

# N random lines in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(LineSegment{Dim, F}) for i ∈ 1:N ]
end

