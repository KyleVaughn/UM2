# Triangle in 2D defined by its 3 vertices.

struct Triangle_2D{T <: AbstractFloat} <: Face_2D{T}
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
function (tri::Triangle_2D{T})(r::R, s::S) where {T <: AbstractFloat, 
                                                  R <: Real, 
                                                  S <: Real}
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
    # 2D cross product returns a scalar
    return abs(u⃗ × v⃗)/2
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

function intersect(l::LineSegment_2D{T}, tri::Triangle_2D{T}) where {T <: AbstractFloat}
    # Create the 3 line segments that make up the triangle and intersect each one
    line_segments = (LineSegment_2D(tri.points[1], tri.points[2]),
                     LineSegment_2D(tri.points[2], tri.points[3]),
                     LineSegment_2D(tri.points[3], tri.points[1]))
    intersections = l .∩ line_segments
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 0)
    ipoints = 0x00
    # We need to account for 3 points returned due to vertex intersection
    for (npoints, points) in intersections
        if npoints === 0x01
            if ipoints === 0x00
                p₁ = points[1]
                ipoints = 0x01
            elseif ipoints === 0x01 && (points[1] ≉ p₁)
                p₂ = points[1]
                ipoints = 0x02
            end
        end
    end
    return ipoints, (p₁, p₂)
end
intersect(tri::Triangle_2D, l::LineSegment_2D) = intersect(l, tri)

function Base.show(io::IO, tri::Triangle_2D{T}) where {T <: AbstractFloat}
    println(io, "Triangle_2D{$T}(")
    for i = 1:3
        p = tri.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end
