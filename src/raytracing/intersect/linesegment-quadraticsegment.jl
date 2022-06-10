# The quadratic segment: 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# 𝘂 = 2(𝘅₁ + 𝘅₂ - 2𝘅₃) and 𝘃 = -(3𝘅₁ + 𝘅₂ - 4𝘅₃)
# The line segment: 𝗹(s) = 𝘅₄ + s𝘄
# 𝘅₄ + s𝘄 = r²𝘂 + r𝘃 + 𝘅₁
# s𝘄 = r²𝘂 + r𝘃 + (𝘅₁ - 𝘅₄)
# 𝟬 = r²(𝘂 × 𝘄) + r(𝘃 × 𝘄) + (𝘅₁ - 𝘅₄) × 𝘄
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝘂 × 𝘄)ₖ, b = (𝘃 × 𝘄)ₖ, c = ([𝘅₁ - 𝘅₄] × 𝘄)ₖ
# 0 = ar² + br + c
# If a = 0 
#   r = -c/b
# else
#   r = (-b ± √(b²-4ac))/2a
# We must also solve for s
# 𝘅₄ + s𝘄 = 𝗾(r)
# s𝘄 = 𝗾(r) - 𝘅₄
# s = ([𝗾(r) - 𝘅₄] ⋅𝘄 )/(𝘄 ⋅ 𝘄)
#
# r is invalid if:
#   1) b² < 4ac
#   2) r ∉ [0, 1]   (Curve intersects, segment doesn't)
# s is invalid if:
#   1) s ∉ [0, 1]   (Line intersects, segment doesn't)
function Base.intersect(l::LineSegment{Point{2,T}}, 
                        q::QuadraticSegment{Point{2,T}}) where {T}
    pmiss = Point{2,T}(INF_POINT,INF_POINT)
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
        return valid ? Vec(l(r), pmiss) : Vec(pmiss, pmiss)
    else
        𝘂 = 2𝘃₁₂ - 4𝘃₁₃ 
        𝘃 = 4𝘃₁₃ - 𝘃₁₂  
        𝘄 = l[2] - l[1]
        a = 𝘂 × 𝘄 
        b = 𝘃 × 𝘄
        c = (q[1] - l[1]) × 𝘄
        w² = 𝘄  ⋅ 𝘄  # 0 ≤ w² 
        if a == 0
            r = -c/b
            0 ≤ r ≤ 1 || return Vec(pmiss, pmiss) 
            p = q(r)
            s = (p - l[1]) ⋅ 𝘄 
            # Since 0 ≤ w², we may test 0 ≤ s ≤ w², and avoid a division by
            # w² in computing s
            return 0 ≤ s && s ≤ w² ? Vec(p, pmiss) : Vec(pmiss, pmiss)
        elseif b^2 ≥ 4a*c
            r₁ = (-b - √(b^2 - 4a*c))/2a
            r₂ = (-b + √(b^2 - 4a*c))/2a
            p₁ = pmiss
            p₂ = pmiss
            if 0 ≤ r₁ ≤ 1
                x = q(r₁)
                s₁ = (x - l[1])⋅𝘄
                if 0 ≤ s₁ && s₁ ≤ w²
                    p₁ = x
                end
            end
            if 0 ≤ r₂ ≤ 1
                y = q(r₂)
                s₂ = (y - l[1])⋅𝘄 
                if 0 ≤ s₂ && s₂ ≤ w²
                    p₂ = y
                end
            end
            return Vec(p₁, p₂)
        else
            return Vec(pmiss, pmiss)
        end
    end
end
