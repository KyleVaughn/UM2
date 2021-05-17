# A quadratic segment in 3D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃
# The assumed relation of the points may be seen in the diagram below:
#                 ___x⃗₃___
#            ____/        \____
#        ___/                  \___
#     __/                          \__
#   _/                                \__
#  /                                     \
# x⃗₁--------------------------------------x⃗₂
#
# NOTE: x⃗₃ is between x⃗₁ and x⃗₂
#
# Let u⃗ = x⃗₂-x⃗₁. Then the parametric representation of the vector from x⃗₁ to x⃗₂
# is u⃗(t) = x⃗₁ + tu⃗ , with t ∈ [0, 1].
#
# The parametric representation of the quadratic curve is
# q(t) = (a|tu⃗|² + b|tu⃗|)ŷ + tu⃗ + x⃗₁
# similar to the familiar y(x) = ax² + bx + c, where ŷ is the unit vector in the same plane as
# x⃗₁, x⃗₂, and x⃗₃, such that ŷ ⟂ u⃗ and is pointing towards x⃗₃.
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
struct QuadraticSegment{T <: AbstractFloat}
    x⃗::NTuple{3,Point{T}}
    a::T
    b::T
    u⃗::Point{T}
    ŷ::Point{T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
function QuadraticSegment(x⃗₁::Point{T}, x⃗₂::Point{T}, x⃗₃::Point{T}) where {T <: AbstractFloat}
    # Using q(1) = x⃗₂ gives b = -a|u⃗|.
    # Using q(t₃) = x⃗₃, the following steps may be used to derive a
    #   1) v⃗ = x⃗₃ - x⃗₁
    #   2) b = -a|u⃗|
    #   3) × u⃗ both sides, and u⃗ × u⃗ = 0⃗
    #   4) |t₃u⃗| = u⃗ ⋅v⃗/|u⃗|
    #   5) |u⃗|² = u⃗ ⋅u⃗
    #   6) v⃗ × u⃗ = -v⃗ × u⃗
    #   the result:
    #
    #             -(u⃗ ⋅ u⃗) (v⃗ × u⃗) ⋅ (v⃗ × u⃗)
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
        a = T(0)
        b = T(0)
        ŷ = zero(v⃗)
    else
        ŷ = -(v⃗ × u⃗) × u⃗/norm((v⃗ × u⃗) × u⃗)
        a = ( (u⃗ ⋅ u⃗) * (v⃗ × u⃗) ⋅(v⃗ × u⃗) )/( (u⃗ ⋅v⃗)*((u⃗ ⋅ v⃗) - (u⃗ ⋅ u⃗)) * ((ŷ × u⃗) ⋅ (v⃗ × u⃗)) )
        b = -a*norm(u⃗)
    end
    return QuadraticSegment((x⃗₁, x⃗₂, x⃗₃), a, b, u⃗, ŷ)
end

# Base methods
# -------------------------------------------------------------------------------------------------


# Methods
# -------------------------------------------------------------------------------------------------
function (q::QuadraticSegment)(t)
    u⃗ = q.x⃗[2] - q.x⃗[1]
    return (q.a*norm(t*u⃗)^2 + q.b*norm(t*u⃗))*q.ŷ + t*u⃗ + q.x⃗[1]
end
function intersects(l::LineSegment, q::QuadraticSegment)
    # q(t) = (a|tu⃗|² + b|tu⃗|)ŷ + tu⃗ + x⃗₁
    # l(s) = x⃗₄ + sw⃗
    # If a|u⃗|²ŷ × w⃗ ≢ 0⃗
    #   x⃗₄ + sw⃗ = (a|tu⃗|² + b|tu⃗|)ŷ + tu⃗ + x⃗₁
    #   sw⃗ = (a|tu⃗|² + b|tu⃗|)ŷ + tu⃗ + (x⃗₁ - x⃗₄)
    #   For valid t (t ∈ [0,1])
    #   0⃗ = (a|u⃗|²ŷ × w⃗)t² + ((b|u⃗|ŷ + u⃗) × w⃗)t + (x⃗₁ - x⃗₄) × w⃗
    #   A⃗ = (a|u⃗|²ŷ × w⃗), B⃗ = ((b|u⃗|ŷ + u⃗) × w⃗), C⃗ = (x⃗₁ - x⃗₄) × w⃗
    #   0⃗ = t²A⃗ + tB⃗ + C⃗
    #   0 = (A⃗ ⋅ A⃗)t² + (B⃗ ⋅ A⃗)t + (C⃗ ⋅ A⃗)
    #   A = (A⃗ ⋅ A⃗), B = (B⃗ ⋅ A⃗), C = (C⃗ ⋅ A⃗)
    #   0 = At² + Bt + B
    #   t = (-B - √(B²-4AC))/2A, -B + √(B²-4AC))/2A)
    #   s = ((q(t) - x⃗₄)⋅w⃗/(w⃗ ⋅ w⃗)
    #   t is invalid if:
    #     1) A = 0            
    #     2) B² < 4AC       
    #     3) t < 0 or 1 < t   (Line intersects, segment doesn't)
    #   s is invalid if:
    #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
    # If A = 0, we need to use line intersection instead.
    bool = false
    npoints = 0
    type = typeof(l.p⃗₁.coord[1])
    points = [Point(type.((1e9, 1e9, 1e9))), Point(type.((1e9, 1e9, 1e9)))]
    w⃗ = l.p⃗₂ - l.p⃗₁
    A⃗ = q.a*norm(q.u⃗)^2*q.ŷ × w⃗
    B⃗ = (q.b*norm(q.u⃗)*q.ŷ + q.u⃗) × w⃗
    C⃗ = (q.x⃗[1] - l.p⃗₁) × w⃗
    A = A⃗ ⋅ A⃗
    B = B⃗ ⋅ A⃗
    C = C⃗ ⋅ A⃗
    if isapprox(A, type(0), atol = √eps(type))
        # Line intersection
        t = (-C⃗ ⋅ B⃗)/(B⃗ ⋅ B⃗)
        s = (q(t)- l.p⃗₁) ⋅ w⃗/(w⃗ ⋅ w⃗)
        points[1] = q(t)
        if (0.0 ≤ s ≤ 1.0) && (0.0 ≤ t ≤ 1.0)
            bool = true
            npoints = 1
        end
    elseif B^2 ≥ 4*A*C
        # Quadratic intersection
        t⃗ = [(-B - √(B^2-4*A*C))/(2A), (-B + √(B^2-4A*C))/(2A)]
        s⃗ = [(q(t⃗[1]) - l.p⃗₁) ⋅ w⃗/(w⃗ ⋅ w⃗), (q(t⃗[2]) - l.p⃗₁) ⋅ w⃗/(w⃗ ⋅ w⃗)]
        # Check points to see if they are unique, valid intersections.
        for i = 1:2
            p⃗ₜ = q(t⃗[i])
            p⃗ₛ = l(s⃗[i])
            if (0.0 ≤ s⃗[i] ≤ 1.0) && (0.0 ≤ t⃗[i] ≤ 1.0) && !(p⃗ₜ≈ points[1]) && (p⃗ₜ ≈ p⃗ₛ)
                bool = true
                points[npoints + 1] = p⃗ₜ
                npoints += 1
                if npoints == 2
                    break
                end
            end
        end
    end
    return bool, npoints, points
end
intersects(q::QuadraticSegment, l::LineSegment) = intersects(l, q)
