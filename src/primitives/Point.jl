# An N-dimensional point
struct Point{N,T}
    coord::SVector{N,T}
end

const Point_2D = Point{2}
const Point_3D = Point{3}

Base.broadcastable(p::Point) = Ref(p)
Base.@propagate_inbounds function Base.getindex(p::Point, i::Int)
    getfield(p, :coord)[i]
end

# Constructors
# -------------------------------------------------------------------------------------------------
Point{N,T}(x...) where {N,T}= Point{N,T}(SVector{N,T}(x))
Point{N}(x...) where {N}= Point(SVector(x))
Point(x...) = Point(SVector(x))

# Operators
# -------------------------------------------------------------------------------------------------
@inline -(p::Point) = Point(-p.coord)
@inline +(p::Point, n::Number) = Point(p.coord .+ n)
@inline +(n::Number, p::Point) = Point(n .+ p.coord)
@inline -(p::Point, n::Number) = Point(p.coord .- n)
@inline -(n::Number, p::Point) = Point(n .- p.coord)
@inline *(n::Number, p::Point) = Point(n * p.coord) 
@inline *(p::Point, n::Number) = Point(p.coord * n)
@inline /(n::Number, p::Point) = Point(n / p.coord) 
@inline /(p::Point, n::Number) = Point(p.coord / n)
@inline +(p₁::Point, p₂::Point) = p₁.coord + p₂.coord
@inline -(p₁::Point, p₂::Point) = p₁.coord - p₂.coord
@inline ⋅(p₁::Point, p₂::Point) = dot(p₁.coord, p₂.coord)
@inline ×(p₁::Point, p₂::Point) = cross(p₁.coord, p₂.coord)
@inline ==(p::Point, v::Vector) = p.coord == v
@inline ≈(p₁::Point, p₂::Point) = distance²(p₁, p₂) < (1e-5)^2 # 100 nm

# Methods
# -------------------------------------------------------------------------------------------------
@inline distance(p₁::Point, p₂::Point) = norm(p₁ - p₂)
@inline distance²(p₁::Point, p₂::Point) = norm²(p₁ - p₂)
@inline midpoint(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = Point{N,T}((p₁ + p₂)/2)
@inline norm(p::Point) = √(p.coord ⋅ p.coord)
@inline norm²(p::Point) = p.coord ⋅ p.coord

# Sort points based on their distance from a given point
sortpoints(p::Point, points::Vector{<:Point}) = points[sortperm(distance².(Ref(p), points))]
function sortpoints!(p::Point_2D, points::Vector{<:Point_2D})
    permute!(points, sortperm(distance².(Ref(p), points)))
    return nothing
end
