# An N dimensional point 
struct Point{N, R <: Real}
    𝐱::SVector{N, R}
end

const Point_2D{R} = Point{2, R}
const Point_3D{R} = Point{3, R}

# Base
# -------------------------------------------------------------------------------------------------
broadcastable(𝐩::Point) = Ref(𝐩)

# Constructors
# -------------------------------------------------------------------------------------------------
Point_2D(x₁::R, x₂::R) where {R <: Real}= Point_2D{R}(SVector(x₁, x₂))
Point_3D(x₁::R, x₂::R, x₃::R) where {R <: Real}= Point_3D{R}(SVector(x₁, x₂, x₃))

# Operators
# -------------------------------------------------------------------------------------------------
@inline -(𝐩::Point) = Point(-𝐩.𝐱)
@inline +(𝐩::Point, n::Real) = Point(n .+ 𝐩.𝐱)
@inline +(n::Real, 𝐩::Point) = Point(n .+ 𝐩.𝐱)
@inline -(𝐩::Point, n::Real) = Point(𝐩.𝐱 .- n)
@inline -(n::Real, 𝐩::Point) = -(𝐩 - n) 
@inline *(n::Real, 𝐩::Point) = Point(n .* 𝐩.𝐱) 
@inline *(𝐩::Point, n::Real) = Point(n .* 𝐩.𝐱) 
# dot
# cross
# @inline /(𝐩₁::Point_2D, 𝐩₂::Point_2D) = Point_2D(𝐩₁.x/𝐩₂.x, 𝐩₁.y/𝐩₂.y)
 
# # Methods
# # -------------------------------------------------------------------------------------------------
# @inline distance(𝐩₁::Point_2D, 𝐩₂::Point_2D) = norm(𝐩₁ - 𝐩₂)
# @inline distance²(𝐩₁::Point_2D, 𝐩₂::Point_2D) = norm²(𝐩₁ - 𝐩₂)
# @inline isapprox(𝐩₁::Point_2D, 𝐩₂::Point_2D) = distance²(𝐩₁, 𝐩₂) < (5e-6)^2
# @inline midpoint(𝐩₁::Point_2D, 𝐩₂::Point_2D) = (𝐩₁ + 𝐩₂)/2
@inline norm(𝐩::Point) = norm(𝐩.𝐱)
# @inline norm²(𝐩::Point_2D) = 𝐩[1]^2 + 𝐩[2]^2
# 
# # Sort points based on their distance from a given point
# sortpoints(p::Point_2D, points::Vector{<:Point_2D}) = points[sortperm(distance².(p, points))]
# function sortpoints!(p::Point_2D, points::Vector{<:Point_2D})
#     permute!(points, sortperm(distance².(p, points)))
#     return nothing
# end
