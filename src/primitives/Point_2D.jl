# A 2D point in Cartesian coordinates.
struct Point_2D <: FieldVector{2, Float64}
    x::Float64
    y::Float64
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 1D constructor
Point_2D(x::Real) = Point_2D(Float64(x), 0.0)

# Operators (All type-stable)
# -------------------------------------------------------------------------------------------------
≈(p₁::Point_2D, p₂::Point_2D) = distance(p₁, p₂) < Point_2D_differentiation_distance
≉(p₁::Point_2D, p₂::Point_2D) = !(p₁ ≈ p₂)
# Note the cross product of two 2D points returns a scalar. It is assumed that this is the
# desired quantity, since the cross product of vectors in the plane is a vector normal to the plane.
# Hence the z coordinate of the resultant vector is returned.
×(p₁::Point_2D, p₂::Point_2D) = p₁.x*p₂.y - p₂.x*p₁.y
⋅(p₁::Point_2D, p₂::Point_2D) = p₁.x*p₂.x + p₁.y*p₂.y
+(p::Point_2D, n::Real) = p .+ n
+(n::Real, p::Point_2D) = p + n
-(p::Point_2D, n::Real) = p .- n
-(n::Real, p::Point_2D) = p - n
*(n::Real, p::Point_2D) = n .* p
*(p::Point_2D, n::Real) = n * p
/(p::Point_2D, n::Real) = p ./ n

# Methods (All type-stable)
# -------------------------------------------------------------------------------------------------
norm(p::Point_2D) = sqrt(p.x^2 + p.y^2)
distance(p₁::Point_2D, p₂::Point_2D) = norm(p₁ - p₂)
midpoint(p₁::Point_2D, p₂::Point_2D) = (p₁ + p₂)/2

# Sort points based on their distance from a given point
function sort_points(p::Point_2D, points::Vector{Point_2D})
    if 0 < length(points)
        # Sort the points based upon their distance to the point
        return points[sortperm(distance.(Ref(p), points))] 
    else
        return points
    end
end
