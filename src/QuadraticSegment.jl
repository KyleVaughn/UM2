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
# 𝐘(t) = (a|tu⃗|² + b|tu⃗|)ŷ + tu⃗ + x⃗₁
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
    ŷ::Point{T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
function QuadraticSegment(x⃗₁::Point{T}, x⃗₂::Point{T}, x⃗₃::Point{T}) where {T <: AbstractFloat}
    # Using 𝐘(1) = x⃗₂ gives b = -a|u⃗|.
    # Using 𝐘(t₃) = x⃗₃, the following steps may be used to derive a
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
    return QuadraticSegment((x⃗₁, x⃗₂, x⃗₃), a, b, ŷ)
end

# Base methods
# -------------------------------------------------------------------------------------------------
# quad.c gives x1
# evaluate as function of t, need u(t)

# Methods
# -------------------------------------------------------------------------------------------------
