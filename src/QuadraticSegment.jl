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

# Methods
# -------------------------------------------------------------------------------------------------
(q::QuadraticSegment)(t) = t^2*q.r⃗[2] + t*q.r⃗[1] + q.x⃗[1]

function intersects(l::LineSegment, q::QuadraticSegment)
    # q(t) = t²r⃗₂ + tr⃗₁ + r⃗₀ 
    # l(s) = x⃗₄ + sw⃗
    # If r⃗₂ ≢  0⃗
    #   x⃗₄ + sw⃗ = t²r⃗₂ + tr⃗₁ + r⃗₀
    #   sw⃗ = t²r⃗₂ + tr⃗₁ + r⃗₀ - x⃗₄
    #   0⃗ = (t²r⃗₂ + tr⃗₁ + r⃗₀ - x⃗₄) × w⃗
    #   A⃗ = (r⃗₂× w⃗), B⃗ = (r⃗₁× w⃗), C⃗ = (r⃗₀ - x⃗₄) × w⃗ 
    #   0⃗ = t²A⃗ + tB⃗ + C⃗
    #   0 = t²A⃗ᵢ + tB⃗ᵢ + C⃗ᵢ, i = 1, 2, 3
    #   t = (-B⃗ᵢ- √(B⃗ᵢ²-4A⃗ᵢC⃗ᵢ))/2A⃗ᵢ, -B⃗ᵢ+ √(B⃗ᵢ²-4A⃗ᵢC⃗ᵢ))/2A⃗ᵢ)
    #   s = (t²r⃗₂ + tr⃗₁ + r⃗₀ - x⃗₄)⋅w⃗/(w⃗ ⋅ w⃗)
    #   t is invalid if:
    #     1) A⃗ᵢ= 0            (Lines are parallel or colinear)
    #     2) B⃗ᵢ²< 4A⃗ᵢC⃗ᵢ       (No solution. Skip.)
    #     3) t < 0 or 1 < t   (Line intersects, segment doesn't)
    #   s is invalid if:
    #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
    # In r⃗₂ = 0⃗, we need to use line intersection instead. Since A⃗ = (r⃗₂× w⃗) = 0⃗,
    # which would give ±Inf for t, even if the intersection exists.
    # See LineSegment intersects algorithm for derivation.
    bool = false
    npoints = 0
    points = [zero(l.p⃗₁), zero(l.p⃗₁)]
    if q.r⃗[2] ≈ zero(q.r⃗[2])
        # Line intersection
        l₂ = LineSegment(q.x⃗[1], q.x⃗[2])
        bool, points[1] = intersects(l, l₂)
        return bool ? (bool, 1, points) : (bool, 0, points)
    else
        # Quadratic intersection
        # Since we don't know which components of A⃗, B⃗, C⃗ are non-zero, we find both t and s
        # pairs for all three vector components
        type = typeof(l.p⃗₁.coord[1])
        t = type.([-1, -1, -1, -1, -1, -1])
        s = type.([-1, -1, -1, -1, -1, -1])
        w⃗ = l.p⃗₂ - l.p⃗₁
        A⃗ = q.r⃗[2] × w⃗
        B⃗ = q.r⃗[1] × w⃗
        C⃗ = (q.x⃗[1] - l.p⃗₁) × w⃗
        # compute t
        for i = 1:3
            if B⃗[i]^2 ≥ 4A⃗[i]*C⃗[i]
                t[2i-1] = (-B⃗[i] - √(B⃗[i]^2 - 4A⃗[i]*C⃗[i]))/(2A⃗[i])   
                t[2i]   = (-B⃗[i] + √(B⃗[i]^2 - 4A⃗[i]*C⃗[i]))/(2A⃗[i])   
            end
        end
        # compute s
        for i = 1:6
            s[i] = (t[i]^2*q.r⃗[2] + t[i]*q.r⃗[1] + q.x⃗[1] - l.p⃗₁)⋅w⃗/(w⃗ ⋅ w⃗) 
        end
        # find the first two points that satisfy the conditions and are not the same.
        for i = 1:6
            p⃗ₜ = q(t[i])
            p⃗ₛ = l(s[i]) 
            if (0.0 ≤ s[i] ≤ 1.0) && (0.0 ≤ t[i] ≤ 1.0) && (p⃗ₜ ≈ p⃗ₛ) && !(p⃗ₜ≈ points[npoints + 1])
                bool = true
                points[npoints + 1] = p⃗ₜ
                npoints += 1 
            end
        end
        return bool, npoints, points 
    end
end
