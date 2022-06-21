# The quadratic segment: q(r) = P₁ + r𝘂 + r²𝘃
# 𝘃 = 2(P₁ + P₂ - 2P₃) and 𝘂 = -(3P₁ + P₂ - 4P₃)
# The line segment: 𝗹(s) = P₄ + s𝘄
# P₄ + s𝘄 = r²𝘃 + r𝘂 + P₁
# s𝘄 = r²𝘃 + r𝘂 + (P₁ - P₄)
# 𝟬 = r²(𝘃 × 𝘄) + r(𝘂 × 𝘄) + (P₁ - P₄) × 𝘄
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝘃 × 𝘄)ₖ, b = (𝘂 × 𝘄)ₖ, c = ([P₁ - P₄] × 𝘄)ₖ
# 0 = ar² + br + c
# If a = 0 
#   r = -c/b
# else
#   r = (-b ± √(b²-4ac))/2a
# We must also solve for s
# P₄ + s𝘄 = q(r)
# s𝘄 = q(r) - P₄
# s = ([q(r) - P₄] ⋅𝘄 )/(𝘄 ⋅ 𝘄)
#
# r is invalid if:
#   1) b² < 4ac
#   2) r ∉ [0, 1]   (Curve intersects, segment doesn't)
# s is invalid if:
#   1) s ∉ [0, 1]   (Line intersects, segment doesn't)
function Base.intersect(l::LineSegment{Point{2,T}}, 
                        q::QuadraticSegment{Point{2,T}}) where {T}
    P_miss = Point{2,T}(INF_POINT,INF_POINT)
    # Check if the segment is effectively straight.
    # Project P₃ onto the line from P₁ to P₂, call it P₄
    𝘃₁₃ = q[3] - q[1] 
    𝘃₁₂ = q[2] - q[1] 
    v₁₂ = norm²(𝘃₁₂)
    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂)*inv(v₁₂)*𝘃₁₂
    # Determine the distance from P₃ to P₄ (P₄ - P₃ = P₁ + 𝘃₁₄ - P₃ = 𝘃₁₄ - 𝘃₁₃)
    d² = norm²(𝘃₁₄ - 𝘃₁₃)
    if d² < T(EPS_POINT)^2 # Use line segment intersection, segment is effectively straight
        # Line segment intersection looks like the following.
        # We want to reuse quantities we have already computed
        # Here l₁ = l, l₂ = LineSegment(q[1], q[2])
        #    𝘄 = l₂[1] - l₁[1]
        #    𝘂₁= l₁[2] - l₁[1]
        #    𝘂₂= l₂[2] - l₂[1]
        #    z = 𝘂₁ × 𝘂₂
        #    r = (𝘄 × 𝘂₂)/z
        #    s = (𝘄 × 𝘂₁)/z
        #    valid = 0 ≤ r && r ≤ 1 && 0 ≤ s && s ≤ 1
        𝘄 = q[1] - l[1]
        𝘂₁= l[2] - l[1]
        # 𝘂₂= 𝘃₁₂ 
        z = 𝘂₁ × 𝘃₁₂
        r = (𝘄 × 𝘃₁₂)/z
        s = (𝘄 × 𝘂₁)/z
        valid = 0 ≤ r && r ≤ 1 && 0 ≤ s && s ≤ 1
        return valid ? Vec(l(r), P_miss) : Vec(P_miss, P_miss)
    else
        𝘃 = 2𝘃₁₂ - 4𝘃₁₃ 
        𝘂 = 4𝘃₁₃ - 𝘃₁₂  
        𝘄 = l[2] - l[1]
        a = 𝘃 × 𝘄 
        b = 𝘂 × 𝘄
        c = (q[1] - l[1]) × 𝘄
        w² = 𝘄  ⋅ 𝘄  # 0 ≤ w² 
        if a == 0
            r = -c/b
            0 ≤ r ≤ 1 || return Vec(P_miss, P_miss) 
            P = q(r)
            s = (P - l[1]) ⋅ 𝘄 
            # Since 0 ≤ w², we may test 0 ≤ s ≤ w², and avoid a division by
            # w² in computing s
            return 0 ≤ s && s ≤ w² ? Vec(P, P_miss) : Vec(P_miss, P_miss)
        elseif b^2 ≥ 4a*c
            r₁ = (-b - √(b^2 - 4a*c))/2a
            r₂ = (-b + √(b^2 - 4a*c))/2a
            P₁ = P_miss
            P₂ = P_miss
            if 0 ≤ r₁ ≤ 1
                Q₁ = q(r₁)
                s₁ = (Q₁ - l[1])⋅𝘄
                if 0 ≤ s₁ && s₁ ≤ w²
                    P₁ = Q₁
                end
            end
            if 0 ≤ r₂ ≤ 1
                Q₂ = q(r₂)
                s₂ = (Q₂ - l[1])⋅𝘄 
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
