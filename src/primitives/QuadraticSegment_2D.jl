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
# NOTE: x⃗₃ is between x⃗₁ and x⃗₂, but not necessarily the midpoint.
# q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment_2D <: Edge_2D
    points::SVector{3, Point_2D}
end

# Constructors
# -------------------------------------------------------------------------------------------------
QuadraticSegment_2D(p₁::Point_2D, p₂::Point_2D, p₃::Point_2D) = QuadraticSegment_2D(SVector(p₁, p₂, p₃))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(q::QuadraticSegment_2D) = Ref(q)
Base.getindex(q::QuadraticSegment_2D, i::Int64) = q.points[i]
Base.firstindex(q::QuadraticSegment_2D) = 1
Base.lastindex(q::QuadraticSegment_2D) = 3

# Methods (All type-stable)
# -------------------------------------------------------------------------------------------------
# Interpolation
# q(0) = q[1], q(1) = q[2], q(1//2) = q[3]
function (q::QuadraticSegment_2D)(r::Real)
    # See Fhe Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = Float64(r)
    return (2rₜ-1)*(rₜ-1)*q[1] + 
               rₜ*(2rₜ-1)*q[2] + 
               4rₜ*(1-rₜ)*q[3]
end

arc_length(q::QuadraticSegment_2D) = arc_length(q, Val(15)) 

function arc_length(q::QuadraticSegment_2D, ::Val{N}) where {N}
    # This does have an analytic solution, but the Mathematica solution is pages long and can
    # produce NaN results when the segment is straight, so numerical integration is used.
    # (Gauss-Legengre quadrature)
    #     1                  N
    # L = ∫ ||q⃗'(r)||dr  ≈   ∑ wᵢ||q⃗'(rᵢ)||
    #     0                 i=1
    #
    # N is the number of points used in the quadrature.
    # See tuning/QuadraticSegment_2D_arc_length.jl for more info on how N = 15 was chosen
    # as the default value.
    w, r = gauss_legendre_quadrature(Val(N))
    return sum(w .* norm.(derivative.(q, r))) 
end

closest_point(p::Point_2D, q::QuadraticSegment_2D) = closest_point(p, q, 30)

# Return the closest point on the curve to point p and the value of r
# Uses at most N iterations of Newton-Raphson
function closest_point(p::Point_2D, q::QuadraticSegment_2D, N::Int64)
    r = 0.5
    Δr = 0.0
    for i = 1:N
        err = p - q(r)
        D = derivative(q, r)
        if abs(D[1]) > abs(D[2])
            Δr = err[1]/D[1]
        else
            Δr = err[2]/D[2]
        end
        r += Δr
        if abs(Δr) < 1e-7
            break
        end
    end 
    return r, q(r)
end

# Get the derivative dq⃗/dr evalutated at r
function derivative(q::QuadraticSegment_2D, r::Real)
    rₜ = Float64(r)
    return (4rₜ - 3)*q[1] + 
           (4rₜ - 1)*q[2] + 
           (4 - 8rₜ)*q[3]
end

function intersect(l::LineSegment_2D, q::QuadraticSegment_2D)
    # q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
    # q(r) = 2r²(x⃗₁ + x⃗₂ - 2x⃗₃) + r(-3x⃗₁ - x⃗₂ + 4x⃗₃) + x⃗₁
    # Let D⃗ = 2(x⃗₁ + x⃗₂ - 2x⃗₃), E⃗ = (-3x⃗₁ - x⃗₂ + 4x⃗₃), F⃗ = x₁
    # q(r) = r²D⃗ + rE⃗ + F⃗
    # l(s) = x⃗₄ + sw⃗
    # If D⃗ × w⃗ ≠ 0
    #   x⃗₄ + sw⃗ = r²D⃗ + rE⃗ + F⃗
    #   sw⃗ = r²D⃗ + rE⃗ + (F⃗ - x⃗₄)
    #   0 = r²(D⃗ × w⃗) + r(E⃗ × w⃗) + (F⃗ - x⃗₄) × w⃗
    #   Let A = (D⃗ × w⃗), B = (E⃗ × w⃗), C = (F⃗ - x⃗₄) × w⃗
    #   0 = Ar² + Br + C
    #   r = (-B - √(B²-4AC))/2A, -B + √(B²-4AC))/2A)
    #   s = ((q(r) - p₄)⋅w⃗/(w⃗ ⋅ w⃗)
    #   r is invalid if:
    #     1) A = 0
    #     2) B² < 4AC
    #     3) r < 0 or 1 < r   (Curve intersects, segment doesn't)
    #   s is invalid if:
    #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
    # If D⃗ × w⃗ = 0, we need to use line intersection instead.
    #
    # Note that D⃗ is essentially a vector showing twice the displacement of x⃗₃ from the
    # midpoint of the linear segment (x⃗₁, x⃗₂). So, if |D⃗| = 0, the segment is linear.
    ϵ = parametric_coordinate_ϵ
    ϵ₁ = QuadraticSegment_2D_1_intersection_ϵ
    npoints = 0x00000000
    p₁ = Point_2D(0, 0)
    p₂ = Point_2D(0, 0)
    D⃗ = 2*(q[1] + q[2] - 2*q[3])
    E⃗ = 4*q[3] - 3*q[1] - q[2]
    w⃗ = l[2] - l[1]
    A = D⃗ × w⃗
    B = E⃗ × w⃗
    C = (q[1] - l[1]) × w⃗
    w = w⃗ ⋅ w⃗
    if A^2 < w * ϵ₁^2
        # Line intersection
        # Can B = 0 if A = 0 for non-trivial x?
        r = -C/B
        if r < -ϵ || 1 + ϵ < r
            return 0x00000000, SVector(p₁, p₂)
        end
        p₁ = q(r)
        s = ((p₁ - l[1]) ⋅ w⃗)/w
        if (-ϵ ≤ s ≤ 1 + ϵ)
            return 0x00000001, SVector(p₁, p₂)           
        else
            return 0x00000000, SVector(p₁, p₂)
        end
    elseif B^2 ≥ 4A*C
        # Quadratic intersection
        r₁ = (-B - √(B^2 - 4A*C))/2A
        r₂ = (-B + √(B^2 - 4A*C))/2A
        if (-ϵ ≤ r₁ ≤ 1 + ϵ)
            p = q(r₁)
            s₁ = ((p - l[1]) ⋅ w⃗)/w
            if (-ϵ ≤ s₁ ≤ 1 + ϵ)
                p₁ = p
                npoints += 0x00000001
            end
        end
        if (-ϵ ≤ r₂ ≤ 1 + ϵ)
            p = q(r₂)
            s₂ = ((p - l[1]) ⋅ w⃗)/w
            if (-ϵ ≤ s₂ ≤ 1 + ϵ)
                p₂ = p
                npoints += 0x00000001
            end
        end
        if npoints === 0x00000001 && p₁ === Point_2D(0, 0) 
            p₁ = p₂
        end
    end
    return npoints, SVector(p₁, p₂)
end
intersect(q::QuadraticSegment_2D, l::LineSegment_2D) = intersect(l, q)

# Return if the point is left of the quadratic segment
#   p    ^
#   ^   /
# v⃗ |  / u⃗
#   | / 
#   o
function is_left(p::Point_2D, q::QuadraticSegment_2D)
    # Find the closest point to p on the curve.
    r, p_closest = closest_point(p, q)
    # If the r is invalid, take the closest end point.
    # If r is small or beyond the valid range, just use the second point
    # but we already tested q[2] - q[1] × v⃗ < 0, and since it was not false,
    # it must be true
    if r < 1e-3 || 1 < r
        p_closest = q[2]
    end
    # Vector from curve start to closest point
    u⃗ = p_closest - q[1]
    # Vector from curve start to the point of interest
    return u⃗ × v⃗ > 0
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment_2D)
        rr = LinRange(0, 1, 15)
        points = q.(rr)
        return convert_arguments(LS, points)
    end
    
    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{QuadraticSegment_2D})
        point_sets = [convert_arguments(LS, q) for q in Q]
        return convert_arguments(LS, reduce(vcat, [pset for pset in point_sets]))
    end
end
