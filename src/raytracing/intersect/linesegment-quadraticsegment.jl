# The quadratic segment: q(r) = Pβ + rπ + rΒ²π
# π = 2(Pβ + Pβ - 2Pβ) and π = -(3Pβ + Pβ - 4Pβ)
# The line segment: πΉ(s) = Pβ + sπ
# Pβ + sπ = rΒ²π + rπ + Pβ
# sπ = rΒ²π + rπ + (Pβ - Pβ)
# π¬ = rΒ²(π Γ π) + r(π Γ π) + (Pβ - Pβ) Γ π
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (π Γ π)β, b = (π Γ π)β, c = ([Pβ - Pβ] Γ π)β
# 0 = arΒ² + br + c
# If a = 0 
#   r = -c/b
# else
#   r = (-b Β± β(bΒ²-4ac))/2a
# We must also solve for s
# Pβ + sπ = q(r)
# sπ = q(r) - Pβ
# s = ([q(r) - Pβ] βπ )/(π β π)
#
# r is invalid if:
#   1) bΒ² < 4ac
#   2) r β [0, 1]   (Curve intersects, segment doesn't)
# s is invalid if:
#   1) s β [0, 1]   (Line intersects, segment doesn't)
function Base.intersect(l::LineSegment{Point{2, T}},
                        q::QuadraticSegment{Point{2, T}}) where {T}
    P_miss = Point{2, T}(INF_POINT, INF_POINT)
    # Check if the segment is effectively straight.
    # Project Pβ onto the line from Pβ to Pβ, call it Pβ
    πββ = q[3] - q[1]
    πββ = q[2] - q[1]
    vββ = normΒ²(πββ)
    πββ = (πββ β πββ) * inv(vββ) * πββ
    # Determine the distance from Pβ to Pβ (Pβ - Pβ = Pβ + πββ - Pβ = πββ - πββ)
    dΒ² = normΒ²(πββ - πββ)
    if dΒ² < T(EPS_POINT)^2 # Use line segment intersection, segment is effectively straight
        # Line segment intersection looks like the following.
        # We want to reuse quantities we have already computed
        # Here lβ = l, lβ = LineSegment(q[1], q[2])
        #    π = lβ[1] - lβ[1]
        #    πβ= lβ[2] - lβ[1]
        #    πβ= lβ[2] - lβ[1]
        #    z = πβ Γ πβ
        #    r = (π Γ πβ)/z
        #    s = (π Γ πβ)/z
        #    valid = 0 β€ r && r β€ 1 && 0 β€ s && s β€ 1
        π = q[1] - l[1]
        πβ = l[2] - l[1]
        # πβ= πββ 
        z = πβ Γ πββ
        r = (π Γ πββ) / z
        s = (π Γ πβ) / z
        valid = 0 β€ r && r β€ 1 && 0 β€ s && s β€ 1
        return valid ? Vec(l(r), P_miss) : Vec(P_miss, P_miss)
    else
        π = 2πββ - 4πββ
        π = 4πββ - πββ
        π = l[2] - l[1]
        a = π Γ π
        b = π Γ π
        c = (q[1] - l[1]) Γ π
        wΒ² = π β π  # 0 β€ wΒ² 
        if a == 0
            r = -c / b
            0 β€ r β€ 1 || return Vec(P_miss, P_miss)
            P = q(r)
            s = (P - l[1]) β π
            # Since 0 β€ wΒ², we may test 0 β€ s β€ wΒ², and avoid a division by
            # wΒ² in computing s
            return 0 β€ s && s β€ wΒ² ? Vec(P, P_miss) : Vec(P_miss, P_miss)
        elseif b^2 β₯ 4a * c
            rβ = (-b - β(b^2 - 4a * c)) / 2a
            rβ = (-b + β(b^2 - 4a * c)) / 2a
            Pβ = P_miss
            Pβ = P_miss
            if 0 β€ rβ β€ 1
                Qβ = q(rβ)
                sβ = (Qβ - l[1]) β π
                if 0 β€ sβ && sβ β€ wΒ²
                    Pβ = Qβ
                end
            end
            if 0 β€ rβ β€ 1
                Qβ = q(rβ)
                sβ = (Qβ - l[1]) β π
                if 0 β€ sβ && sβ β€ wΒ²
                    Pβ = Qβ
                end
            end
            return Vec(Pβ, Pβ)
        else
            return Vec(P_miss, P_miss)
        end
    end
end
