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

function gauss_legendre_quadrature(q::QuadraticSegment{T}, N::Int64) where {T <: AbstractFloat}
    # Default case first.
    if N == 20
        weights = T.([
                        0.00880700356957606,
                        0.02030071490019347,
                        0.03133602416705453,
                        0.04163837078835238,
                        0.0509650599086202,
                        0.0590972659807592,
                        0.0658443192245883,
                        0.07104805465919105,
                        0.07458649323630186,
                        0.07637669356536295,
                        0.07637669356536295,
                        0.07458649323630186,
                        0.07104805465919105,
                        0.0658443192245883,
                        0.0590972659807592,
                        0.0509650599086202,
                        0.04163837078835238,
                        0.03133602416705453,
                        0.02030071490019347,
                        0.00880700356957606
                    ])
        r = T.([
                0.0034357004074525577,
                0.018014036361043095,
                0.04388278587433703,
                0.08044151408889061,
                0.1268340467699246,
                0.1819731596367425,
                0.24456649902458644,
                0.3131469556422902,
                0.38610707442917747,
                0.46173673943325133,
                0.5382632605667487,
                0.6138929255708225,
                0.6868530443577098,
                0.7554335009754136,
                0.8180268403632576,
                0.8731659532300754,
                0.9195584859111094,
                0.956117214125663,
                0.981985963638957,
                0.9965642995925474
               ])
    elseif N == 1
        weights = T.([1])
        r = T.([0.5])
    elseif N == 2
        weights = T.([0.5, 0.5])
        r = T.([0.21132486540518713, 0.7886751345948129])
    elseif N == 3
        weights = T.([
                        0.2777777777777776,
                        0.4444444444444444,
                        0.2777777777777776
                     ])
        r = T.([
                0.1127016653792583,
                0.5,
                0.8872983346207417
               ])
    elseif N == 4
        weights = T.([
                        0.17392742256872684,
                        0.3260725774312732,
                        0.3260725774312732,
                        0.17392742256872684
                    ])
        r = T.([
                0.06943184420297371,
                0.33000947820757187,
                0.6699905217924281,
                0.9305681557970262
               ])
    elseif N == 5
        weights = T.([
                        0.1184634425280945,
                        0.23931433524968326,
                        0.28444444444444444,
                        0.23931433524968326,
                        0.1184634425280945
                    ])
        r = T.([
                0.04691007703066802,
                0.23076534494715845,
                0.5,
                0.7692346550528415,
                0.9530899229693319,
                ])
    elseif N == 10
        weights = T.([
                        0.03333567215434387,
                        0.07472567457529025,
                        0.1095431812579911,
                        0.1346333596549981,
                        0.14776211235737646,
                        0.14776211235737646,
                        0.1346333596549981,
                        0.1095431812579911,
                        0.07472567457529025,
                        0.03333567215434387
                    ])
        r = T.([
                0.013046735741414184,
                0.06746831665550773,
                0.16029521585048778,
                0.2833023029353764,
                0.4255628305091844,
                0.5744371694908156,
                0.7166976970646236,
                0.8397047841495122,
                0.9325316833444923,
                0.9869532642585859
                ])
    else
        weights = T[]
        r= T[]
    end
    return weights, r
end

function derivative(q::QuadraticSegment{T}, r::T) where {T <: AbstractFloat}
    # Just dq/dr
    return (4r - 3)*q.points[1] + (4r - 1)*q.points[2] + (4 - 8r)*q.points[3]
end

function arc_length(q::QuadraticSegment{T}; N::Int64=20) where {T <: AbstractFloat}
    (w, r) = gauss_legendre_quadrature(q, N)
    return sum(norm.(w .* derivative.(q, r)))
end
