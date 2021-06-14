# Quadrilateral defined by its 4 vertices.

# NOTE: Quadrilaterals are assumed to be convex and planar.
# Quadrilateral_2Ds must be convex to be valid finite elements. 
struct Quadrilateral_2D{T <: AbstractFloat}
    # Counter clockwise order
    points::NTuple{4, Point_2D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral_2D(p₁::Point_2D{T}, 
              p₂::Point_2D{T}, 
              p₃::Point_2D{T},
              p₄::Point_2D{T}) where {T <: AbstractFloat} = Quadrilateral_2D((p₁, p₂, p₃, p₄))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad::Quadrilateral_2D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T)*(1 - s_T)*quad.points[1] + 
                 r_T*(1 - s_T)*quad.points[2] + 
                       r_T*s_T*quad.points[3] +
                 (1 - r_T)*s_T*quad.points[4]
end

function triangulate(quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    A, B, C, D = quad.points
    return (Triangle_2D(A, B, C), Triangle_2D(C, D, A))
end

function area(quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    # Using the convex quadrilateral assumption, just return the sum of the areas of the two
    # triangles that partition the quadrilateral. If the convex assumption ever changes, you
    # need to verify that the triangle pair partitions the quadrilateral. Choosing the wrong
    # pair overestimates the area, so just get the areas of both pairs of valid triangles and use
    # the smaller area.
    return sum(area.(triangulate(quad)))
end

function in(p::Point_2D, quad::Quadrilateral_2D)
    return any(p .∈  triangulate(quad))
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, quad::Quadrilateral_2D)
    l₁ = LineSegment_2D(quad.points[1], quad.points[2])
    l₂ = LineSegment_2D(quad.points[2], quad.points[3])
    l₃ = LineSegment_2D(quad.points[3], quad.points[4])
    l₄ = LineSegment_2D(quad.points[4], quad.points[1])
    lines = [l₁, l₂, l₃, l₄]
    return convert_arguments(P, lines)
end

function convert_arguments(P::Type{<:LineSegments}, AQ::AbstractArray{<:Quadrilateral_2D})
    point_sets = [convert_arguments(P, quad) for quad in AQ]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{<:Mesh}, quad::Quadrilateral_2D)
    points = [quad.points[i].coord for i = 1:4]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(P, points, faces)
end

function convert_arguments(MT::Type{Mesh{Tuple{Vector{Quadrilateral_2D{T}}}}},
        AQ::Vector{Quadrilateral_2D{T}}) where {T <: AbstractFloat}
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