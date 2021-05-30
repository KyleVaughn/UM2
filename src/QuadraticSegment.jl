import Base: intersect, in
# A quadratic segment in 3D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃.
# The assumed relation of the points may be seen in the diagram below, where x⃗₃ is not
# necessarily the midpoint of the curve.
#                 ___x⃗₃___
#            ____/        \____
#        ___/                  \___
#     __/                          \__
#   _/                                \__
#  /                                     \
# x⃗₁--------------------------------------x⃗₂
#
# NOTE: x⃗₃ is between x⃗₁ and x⃗₂, but not necessarily the midpoint.
#
# Let u⃗ = x⃗₂-x⃗₁. Then the parametric representation of the vector from x⃗₁ to x⃗₂
# is u⃗(r) = x⃗₁ + ru⃗ , with r ∈ [0, 1].
#
# A general parametric representation of the quadratic curve is:
# q(r) = (a|ru⃗|² + b|ru⃗|)ŷ + ru⃗ + x⃗₁
# similar to the familiar y(x) = ax² + bx + c, where ŷ is the unit vector in the same plane as
# x⃗₁, x⃗₂, and x⃗₃, such that ŷ ⟂ u⃗ and is pointing towards x⃗₃.
#
# We also define v⃗ = x⃗₃-x⃗₁. We see the ŷ vector may be computed by:
# ŷ = -((v⃗ × u⃗) × u⃗)/|(v⃗ × u⃗) × u⃗|
# A diagram of these relations may be seen below:
#                   x⃗₃
#               /
#       v⃗    /      ^
#         /         | ŷ
#      /            |
#   /               |
# x⃗₁--------------------------------------x⃗₂
#                              u⃗
#
# A different parametric representation of the quadratic curve, for which x⃗₃ is the midpoint
# of the curve is:
# q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section

# We use this form over the general form due to the reduced storage requirements and computations
# for many operations. (General formula requires many norm u⃗ operations.)
# However, the general form is used to find the midpoint in the constructor.
struct QuadraticSegment{T <: AbstractFloat} <: Edge
    points::NTuple{3,Point{T}}
    midpoint::Point{T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
function QuadraticSegment(x⃗₁::Point{T}, x⃗₂::Point{T}, x⃗₃::Point{T}) where {T <: AbstractFloat}
    # Using q(1) = x⃗₂ gives b = -a|u⃗|.
    # Using q(r₃) = x⃗₃, the following steps may be used to derive a
    #   1) v⃗ = x⃗₃ - x⃗₁
    #   2) b = -a|u⃗|
    #   3) × u⃗ both sides, and u⃗ × u⃗ = 0⃗
    #   4) |r₃u⃗| = u⃗ ⋅v⃗/|u⃗|
    #   5) |u⃗|² = u⃗ ⋅u⃗
    #   6) v⃗ × u⃗ = -(u⃗ × v⃗)
    #   the result:
    #
    #             (u⃗ ⋅ u⃗) (v⃗ × u⃗) ⋅ (v⃗ × u⃗)
    # a = -------------------------------------------
    #     (u⃗ ⋅ v⃗)[(u⃗ ⋅ v⃗) - (u⃗ ⋅ u⃗)](ŷ × u⃗) ⋅ (v⃗ × u⃗)
    #
    # We can construct ŷ with
    #
    #      -(v⃗ × u⃗) × u⃗
    # ŷ =  -------------
    #      |(v⃗ × u⃗) × u⃗|
    #
    u⃗ = x⃗₂-x⃗₁
    v⃗ = x⃗₃-x⃗₁
    if v⃗ × u⃗ ≈ zero(v⃗)
        # x⃗₃ is on u⃗
        midpoint = x⃗₁ + u⃗/T(2)
    else
        ŷ = -(v⃗ × u⃗) × u⃗/norm((v⃗ × u⃗) × u⃗)
        a = ( (u⃗ ⋅ u⃗) * (v⃗ × u⃗) ⋅(v⃗ × u⃗) )/( (u⃗ ⋅v⃗)*((u⃗ ⋅ v⃗) - (u⃗ ⋅ u⃗) ) * ( (ŷ × u⃗) ⋅ (v⃗ × u⃗)) )
        b = -a*norm(u⃗)
        midpoint = (a*norm(u⃗/T(2))^2 + b*norm(u⃗/T(2)))*ŷ + u⃗/T(2) + x⃗₁
    end
    return QuadraticSegment((x⃗₁, x⃗₂, x⃗₃), midpoint)
end

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
# Points on the curve
function (q::QuadraticSegment)(r::T) where {T <: AbstractFloat}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return (2r-1)*(r-1)*q.points[1] + r*(2r-1)*q.points[2] + 4r*(1-r)*q.midpoint
end

function in(p::Point{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
    # Check to see if a point is on the curve.
    # q is defined by x⃗₁, x⃗₂, and x⃗₃, and the point in question is p⃗
    # Let: 
    #   w⃗ = p⃗ - x⃗₁, 
    #   u⃗ = x⃗₂ - x⃗₁, 
    #   v⃗ = x⃗₃ - x⃗₁
    #        
    #         (u⃗ × v⃗)
    #   n̂ =  ---------  perpendicular to the plane of q 
    #        |(u⃗ × v⃗)|   
    #
    #
    #         n̂ × u⃗   
    #   ŷ =  -------    perpendicular to u⃗ in the plane of q
    #        |n̂ × u⃗|   
    #
    # If p⃗ is in the plane of q, then w⃗ is a linear combination of u⃗ and ŷ, since u⃗ ⟂ ŷ within the plane.
    # w⃗ = ru⃗ + sŷ + tn̂, and t=0. Here we include n̂ to make a square system.
    # Therefore, if A = [u⃗ ŷ n̂] and x = [r; s; t], then Ax=w⃗. 
    # A is invertible since {u⃗, ŷ, n̂} are a basis for R³.
    # If p is actually in the plane of q, then t = 0.
    # If r ∉ (0, 1), then we the point is not in the curve. 
    # If r ∈ (0, 1), we test if q(r) ≈ p
    w⃗ = p - q.points[1]
    u⃗ = q.points[2] - q.points[1]
    v⃗ = q.points[3] - q.points[1]
    n̂ = (u⃗ × v⃗)/norm(u⃗ × v⃗)
    ŷ = (n̂ × u⃗)/norm(n̂ × u⃗)
    if ŷ ≈ 0*ŷ # quadratic is straight
        l = LineSegment(q.points[1], q.points[2])
        return p ∈  l
    else
        A = hcat(u⃗.coord, ŷ.coord, n̂.coord)
        r, s, t = A\w⃗.coord
        return (0 ≤ r ≤ 1) && q(r) ≈ p ? true : false
    end
end

function in_area(p::Point{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
    # Check to see if a point is in the area between u⃗ and the curve.
    # q is defined by x⃗₁, x⃗₂, and x⃗₃, and the point in question is p⃗
    # Let: 
    #   w⃗ = p⃗ - x⃗₁, 
    #   u⃗ = x⃗₂ - x⃗₁, 
    #   v⃗ = x⃗₃ - x⃗₁
    #        
    #         (u⃗ × v⃗)
    #   n̂ =  ---------  perpendicular to the plane of q 
    #        |(u⃗ × v⃗)|   
    #
    #
    #         n̂ × u⃗   
    #   ŷ =  -------    perpendicular to u⃗ in the plane of q
    #        |n̂ × u⃗|   
    #
    # If p⃗ is in the plane of q, then w⃗ is a linear combination of u⃗ and ŷ, since u⃗ ⟂ ŷ within the plane.
    # w⃗ = ru⃗ + sŷ + tn̂, and t=0. Here we include n̂ to make a square system.
    # Therefore, if A = [u⃗ ŷ n̂] and x = [r; s; t], then Ax=w⃗. 
    # A is invertible since {u⃗, ŷ, n̂} are a basis for R³.
    # If p is actually in the plane of q, then t = 0.
    # If r ∉ (0, 1), then we the point is not in the curve. 
    # If r ∈ (0, 1), then we need to see if the point is within the quadratic curve area.
    # Let u(r) = x⃗₁ + ru⃗, then if distance(p, u(r)) ≤ distance(u(r), q(r)) and
    # distance(p, q(r)) ≤ distance(u(r), q(r)), then the point is within the quad area.
    w⃗ = p - q.points[1]
    u⃗ = q.points[2] - q.points[1]
    v⃗ = q.points[3] - q.points[1]
    n̂ = (u⃗ × v⃗)/norm(u⃗ × v⃗)
    ŷ = (n̂ × u⃗)/norm(n̂ × u⃗)
    if ŷ ≈ 0*ŷ # quadratic is straight
        l = LineSegment(q.points[1], q.points[2])
        return p ∈  l
    else
        A = hcat(u⃗.coord, ŷ.coord, n̂.coord)
        r, s, t = A\w⃗.coord
        p_q = q(r)
        p_u = q.points[1] + r*u⃗
        return distance(p, p_q) ≤ distance(p_q, p_u) && distance(p, p_u) ≤ distance(p_q, p_u)
    end
end

function intersect(l::LineSegment{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
    # Here x⃗₃ is the midpoint of the curve.
    # q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
    # q(r) = 2r²(x⃗₁ + x⃗₂ - 2x⃗₃) + r(-3x⃗₁ - x⃗₂ + 4x⃗₃) + x⃗₁
    # Let D⃗ = 2(x⃗₁ + x⃗₂ - 2x⃗₃), E⃗ = (-3x⃗₁ - x⃗₂ + 4x⃗₃), F⃗ = x₁ 
    # q(r) = r²D⃗ + rE⃗ + F⃗
    # l(s) = x⃗₄ + sw⃗
    # If D⃗ × w⃗ ≠ 0⃗
    #   x⃗₄ + sw⃗ = r²D⃗ + rE⃗ + F⃗
    #   sw⃗ = r²D⃗ + rE⃗ + (F⃗ - x⃗₄)
    #   0⃗ = r²(D⃗ × w⃗) + r(E⃗ × w⃗) + (F⃗ - x⃗₄) × w⃗
    #   Let A⃗ = (D⃗ × w⃗), B⃗ = (E⃗ × w⃗), C⃗ = (F⃗ - x⃗₄) × w⃗
    #   0⃗ = r²A⃗ + rB⃗ + C⃗
    #   0 = (A⃗ ⋅ A⃗)r² + (B⃗ ⋅ A⃗)r + (C⃗ ⋅ A⃗)
    #   A = (A⃗ ⋅ A⃗), B = (B⃗ ⋅ A⃗), C = (C⃗ ⋅ A⃗)
    #   0 = Ar² + Br + C
    #   r = (-B - √(B²-4AC))/2A, -B + √(B²-4AC))/2A)
    #   s = ((q(r) - p₄)⋅w⃗/(w⃗ ⋅ w⃗)
    #   r is invalid if:
    #     1) A = 0            
    #     2) B² < 4AC       
    #     3) r < 0 or 1 < r   (Line intersects, segment doesn't)
    #   s is invalid if:
    #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
    # If A = 0, we need to use line intersection instead.
    bool = false
    npoints = 0
    points = [Point(T.((1e9, 1e9, 1e9))), Point(T.((1e9, 1e9, 1e9)))]
    D⃗ = 2*(q.points[1] + q.points[2] - 2*q.midpoint)
    E⃗ = 4*q.midpoint - 3*q.points[1] - q.points[2]
    w⃗ = l.points[2] - l.points[1]
    A⃗ = (D⃗ × w⃗)
    B⃗ = (E⃗ × w⃗)
    C⃗ = (q.points[1] - l.points[1]) × w⃗
    A = A⃗ ⋅ A⃗
    B = B⃗ ⋅ A⃗
    C = C⃗ ⋅ A⃗
    if isapprox(A, 0, atol = √eps(T))
        # Line intersection
        r = (-C⃗ ⋅ B⃗)/(B⃗ ⋅ B⃗)
        s = (q(r)- l.points[1]) ⋅ w⃗/(w⃗ ⋅ w⃗)
        points[1] = q(r)
        if (0.0 ≤ s ≤ 1.0) && (0.0 ≤ r ≤ 1.0)
            bool = true
            npoints = 1
        end
    elseif B^2 ≥ 4*A*C
        # Quadratic intersection
        r⃗ = [(-B - √(B^2-4*A*C))/(2A), (-B + √(B^2-4A*C))/(2A)]
        s⃗ = [(q(r⃗[1]) - l.points[1]) ⋅ w⃗/(w⃗ ⋅ w⃗), (q(r⃗[2]) - l.points[1]) ⋅ w⃗/(w⃗ ⋅ w⃗)]
        # Check points to see if they are unique, valid intersections.
        for i = 1:2
            pᵣ = q(r⃗[i])
            pₛ = l(s⃗[i])
            if (0.0 ≤ s⃗[i] ≤ 1.0) && (0.0 ≤ r⃗[i] ≤ 1.0) && !(pᵣ≈ points[1]) && (pᵣ ≈ pₛ)
                bool = true
                points[npoints + 1] = pᵣ
                npoints += 1
            end
        end
    end
    return bool, npoints, points
end
intersect(q::QuadraticSegment, l::LineSegment) = intersect(l, q)

#function is_left(p::Point{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
#    # If the point is within the quad area, we need to reverse the linear result.
#    l = LineSegment(q.points[1], q.points[2])
#    u⃗ = q.points[2] - q.points[1]
#    v⃗ = q.points[3] - q.points[1]
#    n̂ = (u⃗ × v⃗)/norm(u⃗ × v⃗)
#    bool = isleft(p, l, n̂ = n̂)
#    # Point is outside quad area, just use is_left
#    # Point is inside quad area, return opposite of is_left
#    return in_area(p, q) ? !bool : bool
#end
