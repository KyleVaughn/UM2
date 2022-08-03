export isleft

# If the point is left of the line segment in the 2D plane.
@inline function isleft(P::Point{2, T}, l::LineSegment{Point{2, T}}) where {T}
    return 0 ≤ (l[2] - l[1]) × (P - l[1])
end

# If the point is left of the quadratic segment in the 2D plane.
function isleft(P::Point{2, T}, q::QuadraticSegment{Point{2, T}}) where {T}
    # If the segment is straight, we can perform the isleft check with the straight
    # line from the segment's start point to the segment's end point.
    # If this condition isn't met, the segment's curvature must be accounted for.
    # We find the point on the curve q that is nearest point to the point of interest.
    # Call this point q_near. We then perform the isleft check with the tangent vector
    # of q at q_near and the vector from q_near to p, p - q_near.
    P₁ = q[1]
    𝘃₁₃ = q[3] - P₁
    𝘃₁₂ = q[2] - P₁
    𝘄 = P - P₁
    𝘃₂₃ = q[3] - q[2]
    v₁₂ = norm²(𝘃₁₂)
    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂) * inv(v₁₂) * 𝘃₁₂
    d = norm(𝘃₁₄ - 𝘃₁₃)
    # If segment is straight, or outside the bounds of the segment
    if d < EPS_POINT || P ∉ boundingbox(q)
        return 0 ≤ 𝘃₁₂ × 𝘄
    else
        # See nearest_point for an explanation of the math.
        # q(r) = P₁ + r𝘂 + r²𝘃
        𝘂 = 3𝘃₁₃ + 𝘃₂₃
        𝘃 = -2(𝘃₁₃ + 𝘃₂₃)
        # f′(r) = ar³ + br² + cr + d = 0
        a = 4(𝘃 ⋅ 𝘃)
        b = 6(𝘂 ⋅ 𝘃)
        c = 2((𝘂 ⋅ 𝘂) - 2(𝘃 ⋅ 𝘄))
        d = -2(𝘂 ⋅ 𝘄)
        # Lagrange's method
        e₁ = s₀ = -b / a
        e₂ = c / a
        e₃ = -d / a
        A = 2e₁^3 - 9e₁ * e₂ + 27e₃
        B = e₁^2 - 3e₂
        if A^2 - 4B^3 > 0 # one real root
            s₁ = ∛((A + √(A^2 - 4B^3)) / 2)
            if s₁ == 0
                s₂ = s₁
            else
                s₂ = B / s₁
            end
            r = (s₀ + s₁ + s₂) / 3
            return 0 ≤ jacobian(q, r) × (P - q(r))
        else # three real roots
            # t₁ is complex cube root
            t₁ = exp(log((A + √(complex(A^2 - 4B^3))) / 2) / 3)
            if t₁ == 0
                t₂ = t₁
            else
                t₂ = B / t₁
            end
            ζ₁ = Complex{T}(-1 / 2, √3 / 2)
            ζ₂ = conj(ζ₁)
            rr = SVector(real((s₀ + t₁ + t₂)) / 3,
                         real((s₀ + ζ₂ * t₁ + ζ₁ * t₂)) / 3,
                         real((s₀ + ζ₁ * t₁ + ζ₂ * t₂)) / 3)
            minval, index = findmin(distance².(Ref(P), q.(rr)))
            r = rr[index]
            return 0 ≤ jacobian(q, r) × (P - q(r))
        end
    end
end

# Test if a point is in a polygon for 2D points/polygons
function Base.in(P::Point{2, T}, poly::Polygon{N, Point{2, T}}) where {N, T}
    for i in Base.OneTo(N - 1)
        isleft(P, LineSegment(poly[i], poly[i + 1])) || return false
    end
    return isleft(P, LineSegment(poly[N], poly[1]))
end

# Test if a point is in a polygon for 2D points/quadratic polygons
function Base.in(P::Point{2, T}, poly::QuadraticPolygon{N, Point{2, T}}) where {N, T}
    M = N ÷ 2
    for i in Base.OneTo(M - 1)
        isleft(P, QuadraticSegment(poly[i], poly[i + 1], poly[i + M])) || return false
    end
    return isleft(P, QuadraticSegment(poly[M], poly[1], poly[N]))
end
