import Base: intersect

struct Triangle{T <: AbstractFloat} <: Face
    points::NTuple{3, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle(p₁::Point{T}, p₂::Point{T}, p₃::Point{T}) where {T <: AbstractFloat} = Triangle((p₁, p₂, p₃))

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
# Evaluation in Barycentric coordinates
function (tri::Triangle)(r::T, s::T) where {T <: AbstractFloat}
    return (1 - r - s)*tri.points[1] + r*tri.points[2] + s*tri.points[3]
end

function area(tri::Triangle{T}) where {T <: AbstractFloat}
    # A = bh/2
    # Let u⃗ = |v₂ - v₁|, v⃗ = |v₃ - v₁|
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.points[2] - tri.points[1] 
    v⃗ = tri.points[3] - tri.points[1] 
    return norm(u⃗ × v⃗)/T(2)
end

function intersect(l::LineSegment, tri::Triangle)
    # Algorithm is
    # Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection. 
    type = typeof(l.points[1].coord[1])
    p = zero(l.points[1])

    E₁ = tri.points[2] - tri.points[1]
    E₂ = tri.points[3] - tri.points[1]
    T = l.points[1] - tri.points[1]
    D = l.points[2] - l.points[1]
    P = D × E₂   
    Q = T × E₁
    det = P ⋅ E₁
    if isapprox(det, 0, atol = sqrt(eps(type))) 
        return false, p
    else
        u = (P ⋅ T)/det
        v = (Q ⋅ D)/det
        t = (Q ⋅ E₂)/det
        return (u < 0) || (v < 0) || (u + v > 1) || (t < 0) || (1 < t) ? (false, p) : (true, l(t))
    end
end
