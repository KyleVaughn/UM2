import Base: intersect

struct Quadrilateral{T <: AbstractFloat} <: Face
    # Counter clockwise order
    vertices::NTuple{4, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral(p₁::Point{T}, 
              p₂::Point{T}, 
              p₃::Point{T},
              p₄::Point{T}) where {T <: AbstractFloat} = Quadrilateral((p₁, p₂, p₃, p₄))

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
function triangulate(quad::Quadrilateral{T}) where {T <: AbstractFloat}
    A, B, C, D = quad.vertices
    tri = (Triangle(A, B, C), Triangle(C, D, A), Triangle(B, C, D), Triangle(D, A, B))
    areas = area.(tri)
    return areas[1] + areas[2] <= areas[3] + areas[4] ? (tri[1], tri[2]) : (tri[1], tri[2])
end

function area(quad::Quadrilateral{T}) where {T <: AbstractFloat}
    A, B, C, D = quad.vertices
    areas = area.((Triangle(A, B, C), Triangle(C, D, A), Triangle(B, C, D), Triangle(D, A, B)))
    A₁ = areas[1] + areas[2]
    A₂ = areas[3] + areas[4]
    return A₁ <= A₂ ? A₁ : A₂
end

function intersect(l::LineSegment, quad::Quadrilateral)
    # Triangulate the quadrilateral, intersect the triangles.
    # In the event that two intersection points are returned due to a planar intersection, return
    # the closest one.
    #
    # Check the line dividing the triangles if no intersection is returned?
    tri = triangulate(quad)
    intersections = l .∩ tri
    bools = (intersections[1][1], intersections[2][1])
    points = (intersections[1][2], intersections[2][2])
    if all(bools)
        distances = (distance(l.p₁, points[1]), distance(l.p₁, points[2]))
        return true, points[argmin(distances)]
    else
        return any(bools), points[argmax(bools)]
    end
end
