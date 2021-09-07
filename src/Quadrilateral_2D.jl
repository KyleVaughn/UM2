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
    # Create the 4 line segments that make up the quadangle and intersect each one
    line_segments = [LineSegment_2D(quad.points[1], quad.points[2]),
                     LineSegment_2D(quad.points[2], quad.points[3]),
                     LineSegment_2D(quad.points[3], quad.points[4]),
                     LineSegment_2D(quad.points[4], quad.points[1])]
    intersections = intersect.(l, line_segments)
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 0)
    have_p₁ = false
    have_p₂ = false
    for (bool, point) in intersections
        if bool
            if !have_p₁
                p₁ = point
                have_p₁ = true   
            elseif !have_p₂   
                p₂ = point
                have_p₂ = true   
            elseif p₁ ≈ p₂ && p₂ ≉ point
                p₂ = point 
            end
        end
    end
    return (have_p₁ && have_p₂), LineSegment_2D(p₁, p₂)
end
