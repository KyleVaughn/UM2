import Base: intersect

struct Triangle6{T <: AbstractFloat} <: Face
    vertices::NTuple{6, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6(p₁::Point{T}, 
         p₂::Point{T}, 
         p₃::Point{T},
         p₄::Point{T},
         p₅::Point{T},
         p₆::Point{T}
        ) where {T <: AbstractFloat} = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------


#function area(tri::Triangle{T}) where {T <: AbstractFloat}
#    # Simply find the area of the quadratic segments and the linear triangle.
#    # Depending on upon if the quadratic point is interior or exterior to the linear triangle,
#    # the area of the quadratic curve is added or subtracted
#    # get from a sign?
#    #
#    # USE POINT IN TRIANGLE!!!
#    #
#    u⃗ = tri.vertices[2] - tri.vertices[1] 
#    v⃗ = tri.vertices[3] - tri.vertices[1] 
#    return norm(u⃗ × v⃗)/T(2)
#end

#function intersect(l::LineSegment, tri::Triangle)
#    # Algorithm is
#    # Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection. 
#    # except modified to work for a line that is coplanar with the triangle.
#    # In the case of a coplanar triangle, the point nearest the
#    type = typeof(l.p₁.coord[1])
#    p = zero(l.p₁)
#
#    E₁ = tri.vertices[2] - tri.vertices[1]
#    E₂ = tri.vertices[3] - tri.vertices[1]
#    T = l.p₁ - tri.vertices[1]
#    D = l.p₂ - l.p₁
#    P = D × E₂   
#    Q = T × E₁
#    det = P ⋅ E₁
#    if isapprox(det, 0, atol = sqrt(eps(type))) 
#        edges = (LineSegment(tri.vertices[1], tri.vertices[2]),
#                 LineSegment(tri.vertices[1], tri.vertices[3]),
#                 LineSegment(tri.vertices[2], tri.vertices[3]))
#        bools = [false, false, false]
#        points = [p, p, p]
#        distances = [type(1e9), type(1e9), type(1e9)]
#        for i = 1:3
#            bools[i], points[i] = l ∩ edges[i]
#            if bools[i]
#                distances[i] = distance(l.p₁, points[i])
#            end
#        end
#        # Give the intersection point closest to the line origin
#        return any(bools), points[argmin(distances)]
#    else
#        u = (P ⋅ T)/det
#        v = (Q ⋅ D)/det
#        t = (Q ⋅ E₂)/det
#        return (u < 0) || (v < 0) || (u + v > 1) || (t < 0) || (1 < t) ? (false, p) : (true, l(t))
#    end
#end
