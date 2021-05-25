import Base: intersect

struct Triangle{T <: AbstractFloat} <: Face
    vertices::NTuple{3, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle(p₁::Point{T}, p₂::Point{T}, p₃::Point{T}) where {T <: AbstractFloat} = Triangle((p₁, p₂, p₃))

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
# Evaluation in Barycentric coordinates
function (tri::Triangle)(u::T, v::T) where {T <: AbstractFloat}
    return (T(1) - u - v)*tri.vertices[1] + u*tri.vertices[2] + v*tri.vertices[3]
end

function area(tri::Triangle{T}) where {T <: AbstractFloat}
    # A = bh/2
    # Let u⃗ = |v₂ - v₁|, v⃗ = |v₃ - v₁|
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.vertices[2] - tri.vertices[1] 
    v⃗ = tri.vertices[3] - tri.vertices[1] 
    return norm(u⃗ × v⃗)/T(2)
end

function intersect(l::LineSegment, tri::Triangle)
    # Algorithm is
    # Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection. 
    # except modified to work for a line that is coplanar with the triangle.
    # In the case of a coplanar triangle, the point nearest the
    type = typeof(l.p₁.coord[1])
    p = zero(l.p₁)

    E₁ = tri.vertices[2] - tri.vertices[1]
    E₂ = tri.vertices[3] - tri.vertices[1]
    T = l.p₁ - tri.vertices[1]
    D = l.p₂ - l.p₁
    P = D × E₂   
    Q = T × E₁
    det = P ⋅ E₁
    if isapprox(det, 0, atol = sqrt(eps(type))) 
        edges = (LineSegment(tri.vertices[1], tri.vertices[2]),
                 LineSegment(tri.vertices[1], tri.vertices[3]),
                 LineSegment(tri.vertices[2], tri.vertices[3]))
        bools = [false, false, false]
        points = [p, p, p]
        distances = [type(1e9), type(1e9), type(1e9)]
        for i = 1:3
            bools[i], points[i] = l ∩ edges[i]
            if bools[i]
                distances[i] = distance(l.p₁, points[i])
            end
        end
        # Give the intersection point closest to the line origin
        return any(bools), points[argmin(distances)]
    else
        u = (P ⋅ T)/det
        v = (Q ⋅ D)/det
        t = (Q ⋅ E₂)/det
        return (u < 0) || (v < 0) || (u + v > 1) || (t < 0) || (1 < t) ? (false, p) : (true, l(t))
    end
end
