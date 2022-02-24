# Intersect
# ---------------------------------------------------------------------------------------------
# Intersection between a line segment and quadratic segment
#
# The quadratic segment: 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
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
function intersect(l::LineSegment2D{T}, q::QuadraticSegment2D{T}) where {T}
    ϵ = T(5e-6) # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
    npoints = 0x0000
    p₁ = nan(Point2D{T})
    p₂ = nan(Point2D{T})
    if isstraight(q) # Use line segment intersection.
        hit, point = LineSegment2D(q[1], q[2]) ∩ l
        if hit
            npoints = 0x0001
        end
        return npoints, SVector(point, p₂)
    else
        𝘂 = q.𝘂 
        𝘃 = q.𝘃 
        𝘄 = l.𝘂
        a = 𝘂 × 𝘄 
        b = 𝘃 × 𝘄
        c = (q.𝘅₁ - l.𝘅₁) × 𝘄
        w² = 𝘄 ⋅ 𝘄 
        if abs(a) < 1e-8
            r = -c/b
            -ϵ ≤ r ≤ 1 + ϵ || return 0x0000, SVector(p₁, p₂)
            p₁ = q(r)
            s = (p₁ - l.𝘅₁)⋅𝘄 
            if -ϵ*w² ≤ s ≤ (1 + ϵ)w²
                npoints = 0x0001
            end
        elseif b^2 ≥ 4a*c
            r₁ = (-b - √(b^2 - 4a*c))/2a
            r₂ = (-b + √(b^2 - 4a*c))/2a
            valid_p₁ = false
            if -ϵ ≤ r₁ ≤ 1 + ϵ
                p₁ = q(r₁)
                s₁ = (p₁ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₁ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                    valid_p₁ = true
                end
            end
            if -ϵ ≤ r₂ ≤ 1 + ϵ
                p₂ = q(r₂)
                s₂ = (p₂ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₂ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                end
            end
            if npoints === 0x0001 && !valid_p₁ 
                p₁ = p₂
            end
        end
        return npoints, SVector(p₁, p₂)
    end
end

