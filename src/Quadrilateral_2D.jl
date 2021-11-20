# Quadrilateral in 2D defined by its 4 vertices.
# NOFE: Quadrilaterals are assumed to be convex. This is because quadrilaterals must be convex
# to be valid finite elements.
# https://math.stackexchange.com/questions/2430691/jacobian-determinant-for-bi-linear-quadrilaterals
struct Quadrilateral_2D{F <: AbstractFloat} <: Face_2D{F}
    # Counter clockwise order
    points::NTuple{4, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral_2D(p₁::Point_2D,
                 p₂::Point_2D,
                 p₃::Point_2D,
                 p₄::Point_2D) = Quadrilateral_2D((p₁, p₂, p₃, p₄))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad::Quadrilateral_2D{F})(r::R, s::S) where {F <: AbstractFloat,
                                                        R <: Real,
                                                        S <: Real}
    # See Fhe Visualization Foolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = F(r)
    sₜ = F(s)
    return (1 - rₜ)*(1 - sₜ)*quad.points[1] +
                 rₜ*(1 - sₜ)*quad.points[2] +
                       rₜ*sₜ*quad.points[3] +
                 (1 - rₜ)*sₜ*quad.points[4]
end

function triangulate(quad::Quadrilateral_2D)
    # Return the two triangles that partition the domain
    A, B, C, D = quad.points
    return (Triangle_2D(A, B, C), Triangle_2D(C, D, A))
end

function area(quad::Quadrilateral_2D{F}) where {F <: AbstractFloat}
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

function intersect(l::LineSegment_2D{F}, quad::Quadrilateral_2D{F}) where {F <: AbstractFloat}
    # Create the 4 line segments that make up the quadrilateral and intersect each one
    line_segments = (LineSegment_2D(quad.points[1], quad.points[2]),
                     LineSegment_2D(quad.points[2], quad.points[3]),
                     LineSegment_2D(quad.points[3], quad.points[4]),
                     LineSegment_2D(quad.points[4], quad.points[1]))
    intersections = l .∩ line_segments
    p₁ = Point_2D(F, 0)
    p₂ = Point_2D(F, 0)
    ipoints = 0x00
    # We need to account for 3 or 4 points returned due to vertex intersection
    for (npoints, point) in intersections
        if npoints === 0x01
            if ipoints === 0x00
                p₁ = point
                ipoints = 0x01
            elseif ipoints === 0x01 && (point ≉ p₁)
                p₂ = point
                ipoints = 0x02
            end
        end
    end
    return ipoints, (p₁, p₂)
end

function Base.show(io::IO, quad::Quadrilateral_2D{F}) where {F <: AbstractFloat}
    println(io, "Quadrilateral_2D{$F}(")
    for i = 1:4
        p = quad.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
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
    points = [quad.points[i].x for i = 1:4]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(P, points, faces)
end

function convert_arguments(MF::Type{Mesh{Tuple{Vector{Quadrilateral_2D{F}}}}},
        AQ::Vector{Quadrilateral_2D{F}}) where {F <: AbstractFloat}
    points = reduce(vcat, [[quad.points[i].x for i = 1:4] for quad in AQ])
    faces = zeros(Int64, 2*length(AQ), 3)
    j = 0
    for i in 1:2:2*length(AQ)
        faces[i    , :] = [1 2 3] + [j j j]
        faces[i + 1, :] = [3 4 1] + [j j j]
        j += 4
    end
    return convert_arguments(MF, points, faces)
end
