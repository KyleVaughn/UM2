# A quadratic segment in 2D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃.
# Fhe assumed relation of the points may be seen in the diagram below.
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
# See Fhe Visualization Foolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment_2D{F <: AbstractFloat} <: Edge_2D{F}
    points::NTuple{3, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
QuadraticSegment_2D(p₁::Point_2D,
                    p₂::Point_2D,
                    p₃::Point_2D) = QuadraticSegment_2D((p₁, p₂, p₃))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(q::QuadraticSegment_2D) = Ref(q)

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# q(0) = q.points[1], q(1) = q.points[2], q(1//2) = q.points[3]
function (q::QuadraticSegment_2D{F})(r::R) where {F <: AbstractFloat, R <: Real}
    # See Fhe Visualization Foolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = F(r)
    return (2rₜ-1)*(rₜ-1)*q.points[1] + rₜ*(2rₜ-1)*q.points[2] + 4rₜ*(1-rₜ)*q.points[3]
end

# Get the derivative dq⃗/dr evalutated at r
function derivative(q::QuadraticSegment_2D{F}, r::R) where {F <: AbstractFloat, R <: Real}
    rₜ = F(r)
    return (4rₜ - 3)*q.points[1] + (4rₜ - 1)*q.points[2] + (4 - 8rₜ)*q.points[3]
end

function arc_length(q::QuadraticSegment_2D{F}; N::Int64=15) where {F <: AbstractFloat}
    # Fhis does have an analytic solution, but the Mathematica solution is pages long and can
    # produce NaN results when the segment is straight, so numerical integration is used.
    # (Gauss-Legengre quadrature)
    #     1                  N
    # L = ∫ ||q⃗'(r)||dr  ≈   ∑ wᵢ||q⃗'(rᵢ)||
    #     0                 i=1
    #
    # N is the number of points used in the quadrature.
    # See tuning/QuadraticSegment_2D_arc_length.jl for more info on how N was chosen.
    w, r = gauss_legendre_quadrature(F, N)
    return sum(w .* norm.(derivative.(q, r)))
end

function intersect(l::LineSegment_2D{F}, q::QuadraticSegment_2D{F}) where {F <: AbstractFloat}
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
    npoints = 0x00
    p₁ = Point_2D(F, 0)
    p₂ = Point_2D(F, 0)
    D⃗ = 2*(q.points[1] + q.points[2] - 2*q.points[3])
    E⃗ = 4*q.points[3] - 3*q.points[1] - q.points[2]
    w⃗ = l.points[2] - l.points[1]
    A = D⃗ × w⃗
    B = E⃗ × w⃗
    C = (q.points[1] - l.points[1]) × w⃗
    if abs(A) < norm(w⃗) * ϵ₁
        # Line intersection
        # Can B = 0 if A = 0 for non-trivial x?
        r = -C/B
        p₁ = q(r)
        s = ((p₁ - l.points[1]) ⋅ w⃗)/(w⃗ ⋅ w⃗)
        if (-ϵ ≤ r ≤ 1 + ϵ) && (-ϵ ≤ s ≤ 1 + ϵ)
            npoints = 0x01
        end
    elseif B^2 ≥ 4A*C
        # Quadratic intersection
        r₁ = (-B - √(B^2-4A*C))/2A
        r₂ = (-B + √(B^2-4A*C))/2A
        p₁ = q(r₁)
        p₂ = q(r₂)
        s₁ = ((p₁ - l.points[1]) ⋅ w⃗)/(w⃗ ⋅ w⃗)
        s₂ = ((p₂ - l.points[1]) ⋅ w⃗)/(w⃗ ⋅ w⃗)

        # Check points to see if they are valid intersections.
        # First r,s valid?
        if (-ϵ ≤ r₁ ≤ 1 + ϵ) && (-ϵ ≤ s₁ ≤ 1 + ϵ)
            npoints = 0x01
        end
        # Second r,s valid?
        if (-ϵ ≤ r₂ ≤ 1 + ϵ) && (-ϵ ≤ s₂ ≤ 1 + ϵ)
            npoints += 0x01
            # If only point 2 is valid, return it in index 1 of points
            if npoints === 0x01
                p₁ = p₂
            end
        end
    end
    return npoints, (p₁, p₂)
end
intersect(q::QuadraticSegment_2D, l::LineSegment_2D) = intersect(l, q)

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment_2D{F}) where {F <: AbstractFloat}
    rr = LinRange{F}(0, 1, 15)
    points = q.(rr)
    coords = reduce(vcat, [[points[i].x, points[i+1].x] for i = 1:length(points)-1])
    return convert_arguments(LS, coords)
end

function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment_2D})
    point_sets = [convert_arguments(P, q) for q in Q]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
end
