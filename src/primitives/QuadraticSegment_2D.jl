# A quadratic segment in 2D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃.
# The assumed relation of the points may be seen in the diagram below.
#                 ___x⃗₃___
#            ____/        \____
#        ___/                  \
#     __/                       x⃗₂
#   _/
#  /
# x⃗₁
#
# NOTE: x⃗₃ is not necessarily the midpoint, or even between x⃗₁ and x⃗₂.
# q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment_2D{F <: AbstractFloat} <: Edge_2D{F}
    points::SVector{3, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
QuadraticSegment_2D(p₁::Point_2D, p₂::Point_2D, p₃::Point_2D) = QuadraticSegment_2D(SVector(p₁, p₂, p₃))

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# q(0) = q[1], q(1) = q[2], q(1//2) = q[3]
function (q::QuadraticSegment_2D{F})(r::Real) where {F <: AbstractFloat}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = F(r)
    return (2rₜ-1)*( rₜ-1)q[1] + rₜ*(2rₜ-1)q[2] + 4rₜ*( 1-rₜ)q[3]
end

arclength(q::QuadraticSegment_2D) = arclength(q, Val(15))
function arclength(q::QuadraticSegment_2D{F}, ::Val{N}) where {F <: AbstractFloat, N}
    # Numerical integration is used.
    # (Gauss-Legengre quadrature)
    #     1                  N
    # L = ∫ ||∇ q⃗(r)||dr  ≈  ∑ wᵢ||∇ q⃗(rᵢ)||
    #     0                 i=1
    #
    w, r = gauss_legendre_quadrature(F, Val(N))
    return sum(@. w * norm(∇(q, r)))
end

# Find the axis-aligned bounding box of the segment.
function boundingbox(q::QuadraticSegment_2D)
    # Find the r coordinates where ∂x/∂r = 0, ∂y/∂r = 0
    # We know ∇ q, so we can directly compute these values
    r_x = (3q[1].x + q[2].x - 4q[3].x)/(4(q[1].x + q[2].x - 2q[3].x))
    if 0 < r_x < 1
        x_extreme = (2r_x-1)*(r_x-1)q[1].x + r_x*(2r_x-1)q[2].x + 4r_x*(1-r_x)q[3].x
        xmin = min(q[1].x, q[2].x, x_extreme)
        xmax = max(q[1].x, q[2].x, x_extreme)
    else
        xmin = min(q[1].x, q[2].x)
        xmax = max(q[1].x, q[2].x)
    end

    r_y = (3q[1].y + q[2].y - 4q[3].y)/(4(q[1].y + q[2].y - 2q[3].y))
    if 0 < r_y < 1
        y_extreme = (2r_y-1)*(r_y-1)q[1].y + r_y*(2r_y-1)q[2].y + 4r_y*(1-r_y)q[3].y
        ymin = min(q[1].y, q[2].y, y_extreme)
        ymax = max(q[1].y, q[2].y, y_extreme)
    else
        ymin = min(q[1].y, q[2].y)
        ymax = max(q[1].y, q[2].y)
    end
    return Rectangle_2D(Point_2D(xmin, ymin), Point_2D(xmax, ymax))
end

# Return the gradient of q, evalutated at r
function gradient(q::QuadraticSegment_2D{F}, r::Real) where {F <: AbstractFloat}
    #∇q = 
    rₜ = F(r)
    return (4rₜ - 3)*(q[1] - q[3]) + (4rₜ - 1)*(q[2] - q[3])
end

# Return if the point is left of the quadratic segment
#   p    ^
#   ^   /
# v⃗ |  / u⃗
#   | /
#   o
# function isleft(p::Point_2D, q::QuadraticSegment_2D)
#     if isstraight(q) || p ∉  boundingbox(q)
#         # We don't need to account for the curve if q is straight or p is outside
#         # q's bounding box
#         u⃗ = q[2] - q[1]
#         v⃗ = p - q[1]
#         return u⃗ × v⃗ > 0
#     else
#         # Get the nearest point on q to p.
#         # Construct vectors from a point on q, close to p_near, to p_near and p. 
#         # Use the cross product of these vectors to determine if p isleft.
#         r, p_near = nearest_point(p, q)
#         
#         if r < 1e-6 || 1 < r # If r is small or beyond the valid range, just use q[2]
#             u⃗ = q[2] - q[1]
#             v⃗ = p - q[1]
#         else # otherwise use a point on q, close to p_near
#             q_base = q(0.95r)
#             u⃗ = p_near - q_base
#             v⃗ = p - q_base
#         end
#         return u⃗ × v⃗ > 0
#     end
# end

# If the quadratic segment is effectively linear
@inline function isstraight(q::QuadraticSegment_2D)
    # u⃗ × v⃗ = |u⃗||v⃗|sinθ
    return abs((q[3] - q[1]) × (q[2] - q[1])) < 1e-8
end

# function intersect(l::LineSegment_2D, q::QuadraticSegment_2D)
#     ϵ = parametric_coordinate_ϵ
#     if isstraight(q) # Use line segment intersection.
#         # See LineSegment_2D for the math behind this.
#         v⃗ = l[2] - l[1]
#         u⃗ = q[2] - q[1]
#         vxu = v⃗ × u⃗ 
#         # Parallel or collinear lines, return.
#         1e-8 < abs(vxu) || return (0x00000000, SVector(Point_2D(), Point_2D()))
#         w⃗ = q[1] - l[1]
#         # Delay division until r,s are verified
#         if 0 <= vxu 
#             lowerbound = (-ϵ)vxu
#             upperbound = (1 + ϵ)vxu
#         else
#             upperbound = (-ϵ)vxu
#             lowerbound = (1 + ϵ)vxu
#         end 
#         r_numerator = w⃗ × u⃗ 
#         s_numerator = w⃗ × v⃗ 
#         if (lowerbound ≤ r_numerator ≤ upperbound) && (lowerbound ≤ s_numerator ≤ upperbound) 
#             return (0x00000001, SVector(l(s_numerator/vxu), Point_2D()))
#         else
#             return (0x00000000, SVector(Point_2D(), Point_2D()))
#         end 
#     else
#         # q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
#         # q(r) = 2r²(x⃗₁ + x⃗₂ - 2x⃗₃) + r(-3x⃗₁ - x⃗₂ + 4x⃗₃) + x⃗₁
#         # Let D⃗ = 2(x⃗₁ + x⃗₂ - 2x⃗₃), E⃗ = (-3x⃗₁ - x⃗₂ + 4x⃗₃), F⃗ = x₁
#         # q(r) = r²D⃗ + rE⃗ + F⃗
#         # l(s) = x⃗₄ + sw⃗
#         # If D⃗ × w⃗ ≠ 0
#         #   x⃗₄ + sw⃗ = r²D⃗ + rE⃗ + F⃗
#         #   sw⃗ = r²D⃗ + rE⃗ + (F⃗ - x⃗₄)
#         #   0 = r²(D⃗ × w⃗) + r(E⃗ × w⃗) + (F⃗ - x⃗₄) × w⃗
#         #   Let A = (D⃗ × w⃗), B = (E⃗ × w⃗), C = (F⃗ - x⃗₄) × w⃗
#         #   0 = Ar² + Br + C
#         #   r = (-B - √(B²-4AC))/2A, -B + √(B²-4AC))/2A)
#         #   s = ((q(r) - p₄)⋅w⃗/(w⃗ ⋅ w⃗)
#         #   r is invalid if:
#         #     1) A = 0
#         #     2) B² < 4AC
#         #     3) r < 0 or 1 < r   (Curve intersects, segment doesn't)
#         #   s is invalid if:
#         #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
#         # If D⃗ × w⃗ = 0, there is only one intersection and the equation reduces to line
#         # intersection.
#         npoints = 0x00000000
#         p₁ = Point_2D()
#         p₂ = Point_2D()
#         D⃗ = 2(q[1] +  q[2] - 2q[3])
#         E⃗ =  4q[3] - 3q[1] -  q[2]
#         w⃗ = l[2] - l[1]
#         A = D⃗ × w⃗
#         B = E⃗ × w⃗
#         C = (q[1] - l[1]) × w⃗
#         w = w⃗ ⋅ w⃗
#         if abs(A) < 1e-8 
#             # Line intersection
#             # Can B = 0 if A = 0 for non-trivial x⃗?
#             r = -C/B
#             (-ϵ ≤ r ≤ 1 + ϵ) || return 0x00000000, SVector(p₁, p₂)
#             p₁ = q(r)
#             s = (p₁ - l[1]) ⋅ w⃗/w
#             if (-ϵ ≤ s ≤ 1 + ϵ)
#                 npoints = 0x00000001
#             end
#         elseif B^2 ≥ 4A*C
#             # Quadratic intersection
#             # The compiler seem seems to catch the √(B^2 - 4A*C), for common subexpression 
#             # elimination, so leaving for readability
#             r₁ = (-B - √(B^2 - 4A*C))/2A
#             r₂ = (-B + √(B^2 - 4A*C))/2A
#             if (-ϵ ≤ r₁ ≤ 1 + ϵ)
#                 p = q(r₁)
#                 s₁ = (p - l[1]) ⋅ w⃗/w
#                 if (-ϵ ≤ s₁ ≤ 1 + ϵ)
#                     p₁ = p
#                     npoints += 0x00000001
#                 end
#             end
#             if (-ϵ ≤ r₂ ≤ 1 + ϵ)
#                 p = q(r₂)
#                 s₂ = (p - l[1]) ⋅ w⃗/w
#                 if (-ϵ ≤ s₂ ≤ 1 + ϵ)
#                     p₂ = p
#                     npoints += 0x00000001
#                 end
#             end
#             if npoints === 0x00000001 && p₁ === Point_2D()
#                 p₁ = p₂
#             end
#         end
#         return npoints, SVector(p₁, p₂)
#     end
# end

# Return the Laplacian of q, evalutated at r
function laplacian(q::QuadraticSegment_2D, r::Real)
    return 4(q[1] + q[2] - 2q[3])
end

nearest_point(p::Point_2D, q::QuadraticSegment_2D) = nearest_point(p, q, 15)
# Return the closest point on the curve to point p and the value of r such that q(r) = p_nearest
# Uses at most N iterations of Newton-Raphson
function nearest_point(p::Point_2D, q::QuadraticSegment_2D, N::Int64)
    r = 0.5
    Δr = 0.0
    for i = 1:N
        err = p - q(r)
        grad = ∇(q, r)
        if abs(grad[1]) > abs(grad[2])
            Δr = err[1]/grad[1]
        else
            Δr = err[2]/grad[2]
        end
        r += Δr
        if abs(Δr) < 1e-7
            break
        end
    end
    return r, q(r)
end

#function newton_raphson(f::Function, J⁻¹::Function, x)
#    xₙ₊₁ = xₙ - J⁻¹(xₙ)f
#end












# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment_2D)
        rr = LinRange(0, 1, 15)
        points = q.(rr)
        coords = reduce(vcat, [[points[i], points[i+1]] for i = 1:length(points)-1])
        return convert_arguments(LS, coords)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment_2D})
        point_sets = [convert_arguments(LS, q) for q in Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
end
