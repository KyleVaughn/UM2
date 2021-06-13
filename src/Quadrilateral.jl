# Quadrilateral defined by its 4 vertices.

# NOTE: Quadrilaterals are assumed to be convex and planar (all points in some plane, not 
# necessarily xy, yx, etc.)!
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
function (quad::Quadrilateral{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T)*(1 - s_T)*quad.points[1] + 
                 r_T*(1 - s_T)*quad.points[2] + 
                       r_T*s_T*quad.points[3] +
                 (1 - r_T)*s_T*quad.points[4]
end

function triangulate(quad::Quadrilateral{T}) where {T <: AbstractFloat}
    A, B, C, D = quad.points
    # Using the convex quadrilateral assumption we can directly
#    tri = (Triangle(A, B, C), Triangle(C, D, A), Triangle(B, C, D), Triangle(D, A, B))
#    areas = area.(tri)
#    return areas[1] + areas[2] <= areas[3] + areas[4] ? (tri[1], tri[2]) : (tri[3], tri[4])
    return (Triangle(A, B, C), Triangle(C, D, A))
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
    return any(p .∈  triangulate(quad))
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, quad::Quadrilateral)
    l₁ = LineSegment(quad.points[1], quad.points[2])
    l₂ = LineSegment(quad.points[2], quad.points[3])
    l₃ = LineSegment(quad.points[3], quad.points[4])
    l₄ = LineSegment(quad.points[4], quad.points[1])
    lines = [l₁, l₂, l₃, l₄]
    return convert_arguments(P, lines)
end

function convert_arguments(P::Type{<:LineSegments}, AQ::AbstractArray{<:Quadrilateral})
    point_sets = [convert_arguments(P, quad) for quad in AQ]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{<:Mesh}, quad::Quadrilateral)
    points = [quad.points[i].coord for i = 1:4]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(P, points, faces)
end

function convert_arguments(MT::Type{Mesh{Tuple{Vector{Quadrilateral{T}}}}},
        AQ::Vector{Quadrilateral{T}}) where {T <: AbstractFloat}
    points = reduce(vcat, [[quad.points[i].coord for i = 1:4] for quad in AQ])
    faces = zeros(Int64, 2*length(AQ), 3)
    j = 0
    for i in 1:2:2*length(AQ)
        faces[i    , :] = [1 2 3] + [j j j]
        faces[i + 1, :] = [3 4 1] + [j j j]
        j += 4
    end
    return convert_arguments(MT, points, faces)
end
