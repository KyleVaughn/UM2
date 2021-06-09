import Base: intersect
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

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(q::QuadraticSegment) = Ref(q)

# Methods
# -------------------------------------------------------------------------------------------------
function (q::QuadraticSegment)(r::T) where {T <: AbstractFloat}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return (2r-1)*(r-1)*q.points[1] + r*(2r-1)*q.points[2] + 4r*(1-r)*q.points[3]
end

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
        if (0 ≤ s ≤ 1) && (0 ≤ r ≤ 1)
            bool = true
            npoints = 1
        end
    elseif B^2 ≥ 4*A*C
        # Quadratic intersection
        r = [(-B - √(B^2-4*A*C))/(2A), (-B + √(B^2-4A*C))/(2A)]
        s = [(q(r[1]) - l.points[1]) ⋅ w⃗/(w⃗ ⋅ w⃗), (q(r[2]) - l.points[1]) ⋅ w⃗/(w⃗ ⋅ w⃗)]
        # Check points to see if they are valid intersections.
        pᵣ = q.(r)
        pₛ = l.(s)
        points = pᵣ
        # First r,s valid?
        if (0 ≤ r[1] ≤ 1) && (0 ≤ s[1] ≤ 1) && (pᵣ[1] ≈ pₛ[1])
            npoints += 1
        end
        # Second r,s valid?
        if (0 ≤ r[2] ≤ 1) && (0 ≤ s[2] ≤ 1) && (pᵣ[2] ≈ pₛ[2])
            npoints += 1
            # If only point 2 is valid, return it in index 1 of points
            if npoints == 1
                points[1] = points[2]
            end
        end
        bool = npoints > 0
    end
    return bool, npoints, points
end
intersect(q::QuadraticSegment, l::LineSegment) = intersect(l, q)

function derivative(q::QuadraticSegment{T}, r::T) where {T <: AbstractFloat}
    # dq⃗/dr
    return (4r - 3)*q.points[1] + (4r - 1)*q.points[2] + (4 - 8r)*q.points[3]
end

function arc_length(q::QuadraticSegment{T}; N::Int64=20) where {T <: AbstractFloat}
    # Mathematica solution is pages long and can produce NaN/∞ results when the segment is
    # straight, so numerical integration is used. (Gauss-Legengre quadrature)
    #     1                  N
    # L = ∫ ||q⃗'(r)||dr  ≈   ∑ wᵢ||q⃗'(rᵢ)||
    #     0                 i=1
    # The default number of points is N = 20, since the timing difference is very small for
    # additional accuracy when compared with N = 15, and small N give poor accuracy.
    w, r = gauss_legendre_quadrature(q, N)
    return sum(norm.(w .* derivative.(q, r)))
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, q::QuadraticSegment{T}) where {T <: AbstractFloat}
    rr = T.(LinRange(0, 1, 50))
    points = q.(rr)
    coords = reduce(vcat, [[points[i].coord, points[i+1].coord] for i = 1:length(points)-1])
    return convert_arguments(P, coords)
end

function convert_arguments(P::Type{<:LineSegments}, AQ::AbstractArray{<:QuadraticSegment})
    point_sets = [convert_arguments(P, q) for q in AQ]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end
