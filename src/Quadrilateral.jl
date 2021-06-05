import Base: intersect, in

# NOTE: Quadrilaterals are assumed to be convex and planar!
# Quadrilaterals must be convex to be valid finite elements. 
# Since quadrilaterals are generated from finite element mesh software, it seems like a good 
# assumption that the software generates valid elements.
struct Quadrilateral{T <: AbstractFloat} <: Face
    # Counter clockwise order
    points::NTuple{4, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral(p₁::Point{T}, 
              p₂::Point{T}, 
              p₃::Point{T},
              p₄::Point{T}) where {T <: AbstractFloat} = Quadrilateral((p₁, p₂, p₃, p₄))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad::Quadrilateral)(r::T, s::T) where {T <: AbstractFloat}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    p = (1 - r)*(1 - s)*quad.points[1] + 
              r*(1 - s)*quad.points[2] + 
                    r*s*quad.points[3] +
              (1 - r)*s*quad.points[4]
    return p
end

function triangulate(quad::Quadrilateral{T}) where {T <: AbstractFloat}
    A, B, C, D = quad.points
    tri = (Triangle(A, B, C), Triangle(C, D, A), Triangle(B, C, D), Triangle(D, A, B))
    areas = area.(tri)
    return areas[1] + areas[2] <= areas[3] + areas[4] ? (tri[1], tri[2]) : (tri[3], tri[4])
end

function area(quad::Quadrilateral{T}) where {T <: AbstractFloat}
    # Using the convex quadrilateral assumption, just return the sum of the areas of the two
    # triangles that partition the quadrilateral. If the convex assumption ever changes, you
    # need to verify that the triangle pair partitions the quadrilateral. Choosing the wrong
    # pair overestimates the area, so just get the areas of both pairs of valid triangles and use
    # the smaller area.
    A, B, C, D = quad.points
    return sum(area.((Triangle(A, B, C), Triangle(C, D, A))))
end

function intersect(l::LineSegment, quad::Quadrilateral)
    # Triangulate the quadrilateral, intersect the triangles.
    tri = triangulate(quad)
    intersections = l .∩ tri
    bools = (intersections[1][1], intersections[2][1])
    points = (intersections[1][2], intersections[2][2])
    return any(bools), points[argmax(bools)]
end

function in(p::Point, quad::Quadrilateral)
    tri = triangulate(quad)
    return any(p .∈ tri)
end
