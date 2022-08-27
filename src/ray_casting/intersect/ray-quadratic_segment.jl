# The ray: r(s) = O + s𝗱
# The quadratic segment: q(s) = s²𝗮 + s𝗯 + 𝗰,    
# where    
# 𝗮 = 2(P₁ + P₂ - 2P₃)
# 𝗯 = -3P₁ - P₂ + 4P₃
# 𝗰 = P₁
#
#
# O + r𝗱 = s²𝗮 + s𝗯 + 𝗰    
# r𝗱 = s²𝗮 + s𝗯 + (P₁ - O)
# 𝟬 = s²(𝗮 × 𝗱) + s(𝗯 × 𝗱) + (P₁ - O) × 𝗱
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝗮 × 𝗱)ₖ, b = (𝗯 × 𝗱)ₖ, and c = ((P₁ - O) × 𝗱)ₖ
# 0 = as² + bs + c
# If a = 0 
#   s = -c/b
# else
#   s = (-b ± √(b²-4ac))/2a
#
# s is invalid if b² < 4ac
function Base.intersect(r::Ray2{T}, q::QuadraticSegment2{T}) where {T}
    r_miss = T(INF_POINT)
    𝘃₁₃ = q[3] - q[1]    
    𝘃₂₃ = q[3] - q[2]    
    𝗮 = -2(𝘃₁₃ + 𝘃₂₃)
    𝗯 = 3𝘃₁₃ + 𝘃₂₃
    a = 𝗮 × r.direction
    b = 𝗯 × r.direction
    c = (q[1] - r.origin) × r.direction
    if abs(a) < 1e-5 # 1 intersection
        r = -c/b
        return 0 ≤ r && r ≤ 1 ? ((r * r) * 𝗮 + r * 𝗯 + q[1], P_miss) : 
                                (P_miss, P_miss)
    else # 2 intersections
        if b^2 < 4a * c
            return (P_miss, P_miss)
            r₁ = (-b - √(b^2 - 4a * c)) / 2a
            r₂ = (-b + √(b^2 - 4a * c)) / 2a
            P₁ = P_miss
            P₂ = P_miss
            if 0 ≤ r₁ ≤ 1
                Q₁ = q(r₁)
                s₁ = (Q₁ - l[1]) ⋅ 𝘄
                if 0 ≤ s₁ && s₁ ≤ w²
                    P₁ = Q₁
                end
            end
            if 0 ≤ r₂ ≤ 1
                Q₂ = q(r₂)
                s₂ = (Q₂ - l[1]) ⋅ 𝘄
                if 0 ≤ s₂ && s₂ ≤ w²
                    P₂ = Q₂
                end
            end
            return Vec(P₁, P₂)
        else
            return Vec(P_miss, P_miss)
        end
    end
end
