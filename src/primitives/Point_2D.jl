# A 2D point in Cartesian coordinates.
struct Point_2D{F <: AbstractFloat} <: FieldVector{2, F}
    x::F
    y::F
end

# Minimum distance between two points to be considered different
const Point_2D_differentiation_distance = 5e-6 # 5e-6 cm

# Base
# -------------------------------------------------------------------------------------------------
broadcastable(p::Point_2D) = Ref(p)
similar_type(::Type{<:Point_2D}, ::Type{F}, s::Size{(2,)}) where {F} = Point_2D{F}

# Operators
# -------------------------------------------------------------------------------------------------
@inline +(p::Point_2D, n::Real) = Point_2D(p.x + n, p.y + n)
@inline +(n::Real, p::Point_2D) = Point_2D(p.x + n, p.y + n)
@inline -(p::Point_2D, n::Real) = Point_2D(p.x - n, p.y - n)
@inline -(n::Real, p::Point_2D) = Point_2D(n - p.x, n - p.y)

# Methods
# -------------------------------------------------------------------------------------------------
@inline distance(p₁::Point_2D, p₂::Point_2D) = norm(p₁ - p₂)
@inline distance²(p₁::Point_2D, p₂::Point_2D) = norm²(p₁ - p₂)
@inline isapprox(p₁::Point_2D, p₂::Point_2D) = distance²(p₁, p₂) < Point_2D_differentiation_distance^2
@inline midpoint(p₁::Point_2D, p₂::Point_2D) = (p₁ + p₂)/2
@inline norm²(p::Point_2D) = p.x^2 + p.y^2

# Sort points based on their distance from a given point
sortpoints(p::Point_2D, points::Vector{<:Point_2D}) = points[sortperm(distance².(p, points))]
function sortpoints!(p::Point_2D, points::Vector{<:Point_2D})
    permute!(points, sortperm(distance².(p, points)))
    return nothing
end
