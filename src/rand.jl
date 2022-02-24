# Random
# ---------------------------------------------------------------------------------------------
Base.rand(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(rand(SVector{Dim, T}))
function Base.rand(::Type{Point{Dim, T}}, num_points::Int64) where {Dim, T}
    return [ Point{Dim, T}(rand(SVector{Dim, T})) for i = 1:num_points ]
end





# Random line in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}) where {Dim, F}
    points = rand(Point{Dim, F}, 2)
    return LineSegment{Dim, F}(points[1], points[2])
end

# N random lines in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(LineSegment{Dim, F}) for i ∈ 1:N ]
end

# Random
# ---------------------------------------------------------------------------------------------
# Random quadratic segment in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}) where {Dim, F} 
    points = rand(Point{Dim, F}, 3)
    return QuadraticSegment(points[1], points[2], points[3])
end

# N random quadratic segments in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(QuadraticSegment{Dim, F}) for i ∈ 1:N ]
end







# Random
# ---------------------------------------------------------------------------------------------
# A random AABox within the Dim-dimensional unit hypercube 
# What does the distribution of AABoxs look like? Is this uniform? 
function Base.rand(::Type{AABox{Dim, T}}) where {Dim, T}
    coord₁ = rand(T, Dim)
    coord₂ = rand(T, Dim)
    return AABox{Dim, T}(Point{Dim, T}(min.(coord₁, coord₂)), 
                         Point{Dim, T}(max.(coord₁, coord₂)))  
end

# N random AABoxs within the Dim-dimensional unit hypercube 
function Base.rand(::Type{AABox{Dim, T}}, num_boxes::Int64) where {Dim, T}
    return [ rand(AABox{Dim, T}) for i ∈ 1:num_boxes ]
end
