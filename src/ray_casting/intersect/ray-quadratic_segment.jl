# The ray: R(r) = O + r𝗱
# The quadratic segment: Q(s) = s²𝗮 + s𝗯 + C,    
# where    
# 𝗮 = 2(P₁ + P₂ - 2P₃)
# 𝗯 = -3P₁ - P₂ + 4P₃
# C = P₁
#
# O + r𝗱 = s²𝗮 + s𝗯 + C    
# r𝗱 = s²𝗮 + s𝗯 + (P₁ - O)
# 𝟬 = s²(𝗮 × 𝗱) + s(𝗯 × 𝗱) + (P₁ - O) × 𝗱
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝗮 × 𝗱)ₖ, b = (𝗯 × 𝗱)ₖ, and c = ((P₁ - O) × 𝗱)ₖ
# 0 = as² + bs + c
# If a = 0 
#   s = -c/b
# else
#   s = (-b ± √(b²-4ac))/2a
# s is invalid if b² < 4ac
# Once we have a valid s, let P = s²𝗮 + s𝗯 + C    
# O + r𝗱 = P ⟹   r = ((P - O) ⋅ 𝗱)/(𝗱 ⋅ 𝗱)
function Base.intersect(R::Ray2{T}, Q::QuadraticSegment2{T}) where {T}
    r_miss = T(INF_POINT)
    𝘃₁₃ = Q[3] - Q[1]    
    𝘃₂₃ = Q[3] - Q[2]    
    𝗮 = -2(𝘃₁₃ + 𝘃₂₃)
    𝗯 = 3𝘃₁₃ + 𝘃₂₃
    C = Q[1]
    𝗱 = R.direction
    d2_inv = 1 / norm2(𝗱)
    O = R.origin
    a = 𝗮 × 𝗱
    b = 𝗯 × 𝗱
    c = (C - O) × 𝗱
    if abs(a) < 1e-5 # 1 intersection
        s = -c/b
        if 0 ≤ s && s ≤ 1
            P = s^2 * 𝗮 + s * 𝗯 + C    
            r = d2_inv * ((P - O) ⋅ 𝗱)
            return (r, r_miss)
        else
            return (r_miss, r_miss)
        end
    else # 2 intersections
        # No valid intersections
        if b^2 < 4 * a * c
            return (r_miss, r_miss)
        end
        r₁ = r_miss
        r₂ = r_miss
        s₁ = (-b - √(b^2 - 4 * a * c)) / 2a
        s₂ = (-b + √(b^2 - 4 * a * c)) / 2a
        if 0 ≤ s₁ && s₁ ≤ 1
            P = s₁^2 * 𝗮 + s₁ * 𝗯 + C    
            r₁ = d2_inv * ((P - O) ⋅ 𝗱) 
        end
        if 0 ≤ s₂ && s₂ ≤ 1
            P = s₂^2 * 𝗮 + s₂ * 𝗯 + C    
            r₂ = d2_inv * ((P - O) ⋅ 𝗱) 
        end
        return (r₁, r₂) 
    end
end
