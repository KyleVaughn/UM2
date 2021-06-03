import Base: intersect, in
# A quadratic segment in 3D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃.
# The assumed relation of the points may be seen in the diagram below.
#                 ___x⃗₃___
#            ____/        \____
#        ___/                  \
#     __/                       x⃗₂   
#   _/                                
#  /                                     
# x⃗₁                                        
#
# NOTE: x⃗₃ is between x⃗₁ and x⃗₂, but not necessarily the midpoint.
# q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment{T <: AbstractFloat} <: Edge
    points::NTuple{3,Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
QuadraticSegment(p₁::Point{T}, 
                 p₂::Point{T}, 
                 p₃::Point{T}) where {T <: AbstractFloat} = QuadraticSegment((p₁, p₂, p₃))

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
# Points on the curve
function (q::QuadraticSegment)(r::T) where {T <: AbstractFloat}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return (2r-1)*(r-1)*q.points[1] + r*(2r-1)*q.points[2] + 4r*(1-r)*q.points[3]
end

# This doesn't work for a segment that has a point3 outside the midpoint
#function in(p::Point{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
#    # Check to see if a point is on the curve.
#    # q is defined by x⃗₁, x⃗₂, and x⃗₃, and the point in question is p⃗
#    # Let: 
#    #   w⃗ = p⃗ - x⃗₁, 
#    #   u⃗ = x⃗₂ - x⃗₁, 
#    #   v⃗ = x⃗₃ - x⃗₁
#    #   n⃗ = u⃗ × v⃗  perpendicular to the plane of q 
#    #   y⃗= n⃗ × u⃗  perpendicular to u⃗ in the plane of q
#    #
#    # If p⃗ is in the plane of q, then w⃗ is a linear combination of u⃗ and y⃗, since u⃗ ⟂ y⃗ within the plane.
#    # w⃗ = ru⃗ + sy⃗ + tn⃗, and t=0. Here we include n⃗ to make a square system.
#    # Therefore, if A = [u⃗ y⃗ n⃗] and x = [r; s; t], then Ax=w⃗. 
#    # A is invertible since {u⃗, y⃗, n⃗} are a basis for R³.
#    # If p is actually in the plane of q, then t = 0.
#    # If r ∉ (0, 1), then we the point is not in the curve. 
#    # If r ∈ (0, 1), we test if q(r) ≈ p
#    # Note that if the quadratic segment is straight, we can simply use the LineSegment test.
#    # We determine if the quadratic is straight by the norm of y⃗
#    w⃗ = p - q.points[1]
#    u⃗ = q.points[2] - q.points[1]
#    v⃗ = q.points[3] - q.points[1]
#    n⃗ = u⃗ × v⃗
#    y⃗ = n⃗ × u⃗
#    if y⃗ ≈ 0*y⃗ # quadratic is straight
#        l = LineSegment(q.points[1], q.points[2])
#        return p ∈  l
#    else
#        A = hcat(u⃗.coord, y⃗.coord, n⃗.coord)
#        r, s, t = A\w⃗.coord
#        return (0 ≤ r ≤ 1) && q(r) ≈ p ? true : false
#    end
#end

function intersect(l::LineSegment{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
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
    D⃗ = 2*(q.points[1] + q.points[2] - 2*q.points[3])
    E⃗ = 4*q.points[3] - 3*q.points[1] - q.points[2]
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

#function AABB(l::LineSegment{T}) where {T <: AbstractFloat}
#    # Axis-aligned bounding box in xy-plane
#    xmin = min(l.points[1][1], l.points[2][1])
#    xmax = max(l.points[1][1], l.points[2][1])
#    ymin = min(l.points[1][2], l.points[2][2])
#    ymax = max(l.points[1][2], l.points[2][2])
#    return (xmin, ymin, xmax, ymax)
#end


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
#function in_area(p::Point{T}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
#    # Check to see if a point is on or between the curve and the line segment 
#    # connectring x⃗₁ and x⃗₂.
#    # q is defined by x⃗₁, x⃗₂, and x⃗₃, and the point in question is p⃗
#    # Let: 
#    #   w⃗ = p⃗ - x⃗₁, 
#    #   u⃗ = x⃗₂ - x⃗₁, 
#    #   v⃗ = x⃗₃ - x⃗₁
#    #   n⃗ = u⃗ × v⃗  perpendicular to the plane of q 
#    #   y⃗= n⃗ × u⃗  perpendicular to u⃗ in the plane of q
#    #
#    # If p⃗ is in the plane of q, then w⃗ is a linear combination of u⃗ and y⃗, since u⃗ ⟂ y⃗ within the plane.
#    # w⃗ = ru⃗ + sy⃗ + tn⃗, and t=0. Here we include n⃗ to make a square system.
#    # Therefore, if A = [u⃗ y⃗ n⃗] and x = [r; s; t], then Ax=w⃗. 
#    # A is invertible since {u⃗, y⃗, n⃗} are a basis for R³.
#    # If p is actually in the plane of q, then t = 0.
#    # If r ∉ (0, 1), then we the point is not in area. 
#    # If r ∈ (0, 1), Let u(r) = x⃗₁ + ru⃗, then if distance(p, u(r)) ≤ distance(u(r), q(r)) and   
#    # distance(p, q(r)) ≤ distance(u(r), q(r)), then the point is within the quad area.   
#    # Note that if the quadratic segment is straight, we can simply use the LineSegment test.
#    # We determine if the quadratic is straight by the norm of y⃗
#    w⃗ = p - q.points[1]
#    u⃗ = q.points[2] - q.points[1]
#    v⃗ = q.points[3] - q.points[1]
#    n⃗ = u⃗ × v⃗
#    y⃗ = n⃗ × u⃗
#    if y⃗ ≈ 0*y⃗ # quadratic is straight
#        l = LineSegment(q.points[1], q.points[2])
#        return p ∈  l
#    else
#        A = hcat(u⃗.coord, y⃗.coord, n⃗.coord)
#        r, s, t = A\w⃗.coord
#        p_q = q(r)
#        p_u = q.points[1] + r*u⃗
#        return distance(p, p_q) ≤ distance(p_q, p_u) && distance(p, p_u) ≤ distance(p_q, p_u)
#    end
#end
