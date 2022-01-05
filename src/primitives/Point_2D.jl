# A 2D point in Cartesian coordinates.
struct Point_2D <: FieldVector{2, Float64}
    x::Float64
    y::Float64
end

# Constructors
# -------------------------------------------------------------------------------------------------
# Use optional args to default to (0, 0)
Point_2D(x::Real=0.0, y::Real=0.0) = Point_2D(Float64(x), Float64(y))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p::Point_2D) = Ref(p)

# Operators
# -------------------------------------------------------------------------------------------------
≈(p₁::Point_2D, p₂::Point_2D) = distance²(p₁, p₂) < Point_2D_differentiation_distance^2
# Note the cross product of two 2D points returns a scalar. It is assumed that this is the
# desired quantity, since the cross product of vectors in the plane is a vector normal to the plane.
# Hence the z coordinate of the resultant vector is returned.
×(p₁::Point_2D, p₂::Point_2D) = p₁.x*p₂.y - p₂.x*p₁.y
⋅(p₁::Point_2D, p₂::Point_2D) = p₁.x*p₂.x + p₁.y*p₂.y
+(p::Point_2D, n::Real) = Point_2D(p.x + n, p.y + n)
+(n::Real, p::Point_2D) = p + n
-(p::Point_2D, n::Real) = Point_2D(p.x - n, p.y - n)
-(n::Real, p::Point_2D) = p - n

# Methods
# -------------------------------------------------------------------------------------------------
norm(p::Point_2D) = sqrt(p.x^2 + p.y^2)
norm²(p::Point_2D) = p.x^2 + p.y^2
distance(p₁::Point_2D, p₂::Point_2D) = norm(p₁ - p₂)
distance²(p₁::Point_2D, p₂::Point_2D) = (p₁ - p₂) ⋅(p₁ - p₂)
midpoint(p₁::Point_2D, p₂::Point_2D) = (p₁ + p₂)/2

# Sort points based on their distance from a given point
function sortpoints(p::Point_2D, points::Vector{Point_2D})
    0 < length(points) ? points[sortperm(distance².(p, points))] : points
end
function sortpoints!(p::Point_2D, points::Vector{Point_2D})
    0 === length(points) || permute!(points, sortperm(distance².(p, points)))
    return nothing
end
