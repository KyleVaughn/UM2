# Triangle in 2D defined by its 3 vertices.

struct Triangle_2D{T <: AbstractFloat}
    points::NTuple{3, Point_2D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle_2D(p₁::Point_2D{T}, 
         p₂::Point_2D{T}, 
         p₃::Point_2D{T}) where {T <: AbstractFloat} = Triangle_2D((p₁, p₂, p₃))

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
function (tri::Triangle_2D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    return (1 - T(r) - T(s))*tri.points[1] + T(r)*tri.points[2] + T(s)*tri.points[3]
end

function area(tri::Triangle_2D{T}) where {T <: AbstractFloat}
    # A = bh/2
    # Let u⃗ = (v₂ - v₁), v⃗ = (v₃ - v₁)
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.points[2] - tri.points[1] 
    v⃗ = tri.points[3] - tri.points[1] 
    # 2D cross product returns a scalar (norm of cross product)
    return (u⃗ × v⃗)/2
end

function in(p::Point_2D{T}, tri::Triangle_2D{T}) where {T <: AbstractFloat}
   # If the point is within the plane of the triangle, then the point is only within the triangle
   # if the areas of the triangles formed by the point and each pair of two vertices sum to the 
   # area of the triangle. Division by 2 is dropped, since it cancels
   # If the vertices are A, B, and C, and the point is P, 
   # P is inside ΔABC iff area(ΔABC) = area(ΔABP) + area(ΔBCP) + area(ΔACP)
   A₁ = abs((tri.points[1] - p) × (tri.points[2] - p))
   A₂ = abs((tri.points[2] - p) × (tri.points[3] - p))
   A₃ = abs((tri.points[3] - p) × (tri.points[1] - p))
   A  = abs((tri.points[2] - tri.points[1]) × (tri.points[3] - tri.points[1]))
   return A₁ + A₂ + A₃ ≈ A
end
