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
# The parametric vector form of a quadratic segment is Q(t) = t²r⃗₂ + tr⃗₁ + r⃗₀
# where t ∈ [0,1]. It is trivial to show that r⃗₀ = x⃗₁, hence only r⃗₁ and r⃗₂ are stored in addition
# to the points x⃗₁, x⃗₂, and x⃗₃
struct QuadraticSegment{T <: AbstractFloat}
    x⃗::NTuple{3,Point{T}} # (x⃗₁, x⃗₂, x⃗₃)
    r⃗::NTuple{2,Point{T}} # (r⃗₁, r⃗₂)
end

# Constructors
# -------------------------------------------------------------------------------------------------
function QuadraticSegment(x⃗₁::Point{T}, x⃗₂::Point{T}, x⃗₃::Point{T}) where {T <: AbstractFloat}
    # Q(t) = t²r⃗₂ + tr⃗₁ + r⃗₀
    # Q(0) = x⃗₁ = r⃗₀
    # Q(1) = x⃗₂ = r⃗₂ + r⃗₁ + x⃗₁
    # Let, u⃗ = x⃗₂-x⃗₁, then
    # u⃗ = r⃗₂ + r⃗₁ ⟹   r⃗₁ = u⃗ - r⃗₂
    # Q(t₃) = x⃗₃ = (t₃)²r⃗₂ + t₃r⃗₁ + x⃗₁
    # Let, v⃗ = x⃗₃-x⃗₁, then
    # v⃗ = (t₃)²r⃗₂ + t₃r⃗₁
    # using r⃗₁ = u⃗ - r⃗₂
    # v⃗ = t₃((t₃-1)r⃗₂ + u⃗)
    # r⃗₂ = (v⃗/t₃ - u⃗)/(t₃ - 1)
    # Let, u⃗(t) = x⃗₁ + tu⃗, then knowing that u⃗(t₃) is equal to the projection of v⃗ onto u⃗,
    # u⃗(t₃) = (v⃗ ⋅ u⃗)/(u⃗ ⋅u⃗) u⃗ = x⃗₁ + t₃u⃗
    # t₃ = ((u⃗ ⋅ v⃗) - (x⃗₁ ⋅ u⃗))/(u⃗ ⋅ u⃗)
    u⃗ = x⃗₂-x⃗₁
    v⃗ = x⃗₃-x⃗₁
    t₃ = ((u⃗ ⋅ v⃗) - (x⃗₁ ⋅ u⃗))/(u⃗ ⋅ u⃗)
    r⃗₂ = (v⃗/t₃ - u⃗)/(t₃ - 1)
    r⃗₁ = u⃗ - r⃗₂
    return QuadraticSegment((x⃗₁, x⃗₂, x⃗₃), (r⃗₁, r⃗₂))
end

# Base methods
# -------------------------------------------------------------------------------------------------
function (q::QuadraticSegment)(t)
    return t^2*q.r⃗[2] + t*q.r⃗[1] + q.x⃗[1]
end
# quad.c gives x1
# evaluate as function of t, need u(t)

# Methods
# -------------------------------------------------------------------------------------------------
