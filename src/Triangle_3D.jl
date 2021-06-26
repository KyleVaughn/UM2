# Triangle in 3D defined by its 3 vertices.

struct Triangle_3D{T <: AbstractFloat}
    points::NTuple{3, Point_3D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle_3D(p₁::Point_3D{T}, 
         p₂::Point_3D{T}, 
         p₃::Point_3D{T}) where {T <: AbstractFloat} = Triangle_3D((p₁, p₂, p₃))

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
function (tri::Triangle_3D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    return (1 - T(r) - T(s))*tri.points[1] + T(r)*tri.points[2] + T(s)*tri.points[3]
end

function area(tri::Triangle_3D{T}) where {T <: AbstractFloat}
    # A = bh/2
    # Let u⃗ = |v₂ - v₁|, v⃗ = |v₃ - v₁|
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.points[2] - tri.points[1] 
    v⃗ = tri.points[3] - tri.points[1] 
    return norm(u⃗ × v⃗)/2
end

function intersect(l::LineSegment_3D{type}, tri::Triangle_3D{type}) where {type <: AbstractFloat}
    # Algorithm is from
    # Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection. 
    p = Point_3D(type, 0)
    E₁ = tri.points[2] - tri.points[1]
    E₂ = tri.points[3] - tri.points[1]
    T = l.points[1] - tri.points[1]
    D = l.points[2] - l.points[1]
    P = D × E₂   
    Q = T × E₁
    det = P ⋅ E₁
    if abs(det) < 5e-6
        return false, p
    else
        u = (P ⋅ T)/det
        v = (Q ⋅ D)/det
        t = (Q ⋅ E₂)/det
        return (u < 0) || (v < 0) || (u + v > 1) || (t < 0) || (1 < t) ? (false, p) : (true, l(t))
    end
end
