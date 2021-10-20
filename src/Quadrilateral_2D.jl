# Quadrilateral in 2D defined by its 4 vertices.

# NOTE: Quadrilaterals are assumed to be convex. This is because quadrilaterals must be convex 
# to be valid finite elements.
# https://math.stackexchange.com/questions/2430691/jacobian-determinant-for-bi-linear-quadrilaterals
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
function (quad::Quadrilateral_2D{T})(r::R, s::S) where {T <: AbstractFloat, 
                                                        R <: Real, 
                                                        S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = T(r)
    sₜ = T(s)
    return (1 - rₜ)*(1 - sₜ)*quad.points[1] + 
                 rₜ*(1 - sₜ)*quad.points[2] + 
                       rₜ*sₜ*quad.points[3] +
                 (1 - rₜ)*sₜ*quad.points[4]
end

function triangulate(quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    # Return the two triangles that partition the domain
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

function in(p::Point_2D{T}, quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    return any(p .∈  triangulate(quad))
end

function intersect(l::LineSegment_2D{T}, quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    # Create the 4 line segments that make up the quadrilateral and intersect each one
    line_segments = (LineSegment_2D(quad.points[1], quad.points[2]),
                     LineSegment_2D(quad.points[2], quad.points[3]),
                     LineSegment_2D(quad.points[3], quad.points[4]),
                     LineSegment_2D(quad.points[4], quad.points[1]))
    intersections = l .∩ line_segments
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 0)
    ipoints = 0
    # We need to account for 3 or 4 points returned due to vertex intersection
    for (npoints, points) in intersections
        if npoints === 1
            if ipoints === 0
                p₁ = points[1]
                ipoints = 1 
            elseif ipoints === 1 && (points[1] ≉ p₁) 
                p₂ = points[1]
                ipoints = 2 
            end
        end
    end 
    # Return points, since the final goal is a vector of points
    # Return 4 points, since this is the max number of intersections for 2D finite elements,
    # meaning all elements have the same return type for intersection.
    return ipoints, (p₁, p₂, p₁, p₂) 
end

function Base.show(io::IO, quad::Quadrilateral_2D{T}) where {T <: AbstractFloat}
    println(io, "Quadrilateral_2D{$T}(")
    for i = 1:4
        p = quad.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end
