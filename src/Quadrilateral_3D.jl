# Quadrilateral in 3D defined by its 4 vertices.

# NOTE: Quadrilaterals are assumed to be convex and planar (all points in some plane, not 
# necessarily xy, yx, etc.)!
# Quadrilaterals must be convex to be valid finite elements. See link below
# https://math.stackexchange.com/questions/2430691/jacobian-determinant-for-bi-linear-quadrilaterals
struct Quadrilateral_3D{T <: AbstractFloat}
    # Counter clockwise order
    points::NTuple{4, Point_3D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral_3D(p₁::Point_3D{T}, 
              p₂::Point_3D{T}, 
              p₃::Point_3D{T},
              p₄::Point_3D{T}) where {T <: AbstractFloat} = Quadrilateral_3D((p₁, p₂, p₃, p₄))

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
function (quad::Quadrilateral_3D{T})(r::R, s::S) where {T <: AbstractFloat, 
                                                        R <: Real,
                                                        S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T)*(1 - s_T)*quad.points[1] + 
                 r_T*(1 - s_T)*quad.points[2] + 
                       r_T*s_T*quad.points[3] +
                 (1 - r_T)*s_T*quad.points[4]
end

function triangulate(quad::Quadrilateral_3D{T}) where {T <: AbstractFloat}
    # Return the two triangles that partition the domain
    A, B, C, D = quad.points
    return (Triangle_3D(A, B, C), Triangle_3D(C, D, A))
end

function area(quad::Quadrilateral_3D{T}) where {T <: AbstractFloat}
    # Using the convex quadrilateral assumption, just return the sum of the areas of the two
    # triangles that partition the quadrilateral. If the convex assumption ever changes, you
    # need to verify that the triangle pair partitions the quadrilateral. Choosing the wrong
    # pair overestimates the area, so just get the areas of both pairs of valid triangles and use
    # the smaller area.
    return sum(area.(triangulate(quad)))
end

function intersect(l::LineSegment_3D{T}, quad::Quadrilateral_3D{T}) where {T <: AbstractFloat}
    # Triangulate the quadrilateral, intersect the triangles.
    tri = triangulate(quad)
    intersection1 = l ∩ tri[1]
    if intersection1[1]
        return true, intersection1[2]
    end
    intersection2 = l ∩ tri[2]
    if intersection2[1]
        return true, intersection2[2]
    end
    return false, Point_3D(T, 0)
end
