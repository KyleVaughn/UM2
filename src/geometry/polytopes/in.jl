export isleft

# If the point is left of the line segment in the 2D plane.
@inline function isleft(P::Point{2, T}, l::LineSegment{Point{2, T}}) where {T}
    return 0 β€ (l[2] - l[1]) Γ (P - l[1])
end

# If the point is left of the quadratic segment in the 2D plane.
function isleft(P::Point{2, T}, q::QuadraticSegment{Point{2, T}}) where {T}
    # If the segment is straight, we can perform the isleft check with the straight
    # line from the segment's start point to the segment's end point.
    # If this condition isn't met, the segment's curvature must be accounted for.
    # We find the point on the curve q that is nearest point to the point of interest.
    # Call this point q_near. We then perform the isleft check with the tangent vector
    # of q at q_near and the vector from q_near to p, p - q_near.
    Pβ = q[1]
    πββ = q[3] - Pβ
    πββ = q[2] - Pβ
    π = P - Pβ
    πββ = q[3] - q[2]
    vββ = normΒ²(πββ)
    πββ = (πββ β πββ) * inv(vββ) * πββ
    d = norm(πββ - πββ)
    # If segment is straight
    if d < EPS_POINT
        return 0 β€ πββ Γ π
    else
        # See nearest_point for an explanation of the math.
        # q(r) = Pβ + rπ + rΒ²π
        π = 3πββ + πββ
        π = -2(πββ + πββ)
        # fβ²(r) = arΒ³ + brΒ² + cr + d = 0
        a = 4(π β π)
        b = 6(π β π)
        c = 2((π β π) - 2(π β π))
        d = -2(π β π)
        # Lagrange's method
        eβ = sβ = -b / a
        eβ = c / a
        eβ = -d / a
        A = 2eβ^3 - 9eβ * eβ + 27eβ
        B = eβ^2 - 3eβ
        if A^2 - 4B^3 > 0 # one real root
            sβ = β((A + β(A^2 - 4B^3)) / 2)
            if sβ == 0
                sβ = sβ
            else
                sβ = B / sβ
            end
            r = (sβ + sβ + sβ) / 3
            return 0 β€ jacobian(q, r) Γ (P - q(r))
        else # three real roots
            # tβ is complex cube root
            tβ = exp(log((A + β(complex(A^2 - 4B^3))) / 2) / 3)
            if tβ == 0
                tβ = tβ
            else
                tβ = B / tβ
            end
            ΞΆβ = Complex{T}(-1 / 2, β3 / 2)
            ΞΆβ = conj(ΞΆβ)
            rr = SVector(real((sβ + tβ + tβ)) / 3,
                         real((sβ + ΞΆβ * tβ + ΞΆβ * tβ)) / 3,
                         real((sβ + ΞΆβ * tβ + ΞΆβ * tβ)) / 3)
            minval, index = findmin(distanceΒ².(Ref(P), q.(rr)))
            r = rr[index]
            return 0 β€ jacobian(q, r) Γ (P - q(r))
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
    M = N Γ· 2
    for i in Base.OneTo(M - 1)
        isleft(P, QuadraticSegment(poly[i], poly[i + 1], poly[i + M])) || return false
    end
    return isleft(P, QuadraticSegment(poly[M], poly[1], poly[N]))
end
