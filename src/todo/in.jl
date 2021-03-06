# Nearest point
# ---------------------------------------------------------------------------------------------
# Find the point on πΎ(r) closest to the point of interest π.
#
# Note: r β [0, 1] is not necessarily true for this function, since it finds the minimizer
# of the function πΎ(r), βr β β
# Find r which minimizes βπ - πΎ(r)β, where πΎ(r) = rΒ²π + rπ + πβ.
# This r also minimizes βπ - πΎ(r)βΒ²
# It can be shown that this is equivalent to finding the minimum of the quartic function
# βπ - πΎ(r)βΒ² = f(r) = aβrβ΄ + aβrΒ³ + aβrΒ² + aβr + aβ
# The minimum of f(r) occurs when fβ²(r) = arΒ³ + brΒ² + cr + d = 0, where
# π = π - πβ, a = 4(π β π), b = 6(π β π), c = 2[(π β π) - 2(π βπ)], d = -2(π β π)
# Lagrange's method (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
# is used to find the roots.
function nearest_point(p::Point{Dim, T}, q::QuadraticSegment) where {Dim, T}
    π = q.π
    π = q.π
    π = p - q.πβ
    # fβ²(r) = arΒ³ + brΒ² + cr + d = 0
    a = 4(π β π)
    b = 6(π β π)
    c = 2((π β π) - 2(π β π))
    d = -2(π β π)
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
        return r, q(r)
    else # three real roots
        # Complex cube root
        tβ = exp(log((A + β(complex(A^2 - 4B^3))) / 2) / 3)
        if tβ == 0
            tβ = tβ
        else
            tβ = B / tβ
        end
        ΞΆβ = Complex{T}(-1 / 2, β3 / 2)
        ΞΆβ = conj(ΞΆβ)
        dist_min = typemax(T)
        r_near = zero(T)
        p_near = nan(Point{Dim, T})
        # Use the real part of each root
        for rα΅’ in (real((sβ + tβ + tβ) / 3),
                   real((sβ + ΞΆβ * tβ + ΞΆβ * tβ) / 3),
                   real((sβ + ΞΆβ * tβ + ΞΆβ * tβ) / 3))
            pα΅’ = q(rα΅’)
            dist = distanceΒ²(pα΅’, p)
            if dist < dist_min
                dist_min = dist
                r_near = rα΅’
                p_near = pα΅’
            end
        end
        return r_near, p_near
    end
end

# Hyperplane 
# ---------------------------------------------------------------------------------------------
Base.in(p::Point, plane::Hyperplane) = p.coord β plane.π» β plane.d
in_halfspace(p::Point, plane::Hyperplane) = p.coord β plane.π» - plane.d β₯ 0

# Given a point p and line l that lie in the plane. Check that the point is left of the line
function isleft(p::Point3D, l::LineSegment3D, plane::Hyperplane3D)
    # Since p and l β plane, l.π Γ (p - l.πβ) must either by in the exact same direction
    # as plane.π» or the exact opposite direction. If the direction is the same, the point
    # is left of the line.
    return 0 β€ (l.π Γ (p - l.πβ)) β plane.π»
end

# AABox 
# ---------------------------------------------------------------------------------------------
@inline function Base.in(p::Point2D, aab::AABox2D)
    return aab.xmin β€ p[1] β€ aab.xmax &&
           aab.ymin β€ p[2] β€ aab.ymax
end
@inline function Base.in(p::Point3D, aab::AABox3D)
    return aab.xmin β€ p[1] β€ aab.xmax &&
           aab.ymin β€ p[2] β€ aab.ymax &&
           aab.zmin β€ p[3] β€ aab.zmax
end
# Point inside polygon
# ---------------------------------------------------------------------------------------------
# Not necessarily planar
#function Base.in(p::Point3D, poly::Polygon{N, 3}) where {N}
#    # Check if the point is even in the same plane as the polygon
#    plane = Hyperplane(poly[1], poly[2], poly[3])
#    p β plane || return false
#    # Test that the point is to the left of each edge, oriented to the plane
#    for i = 1:N-1
#        isleft(p, LineSegment3D(poly[i], poly[i + 1]), plane) || return false
#    end
#    return isleft(p, LineSegment3D(poly[N], poly[1]), plane) 
#end
#

# Point inside triangle 
# ---------------------------------------------------------------------------------------------
function Base.in(p::Point3D, tri::Triangle3D)
    # P β ABC iff the surface normals of CCW triangles PAB, PBC, & PCA are equal.
    π? = tri[1] - p
    π― = tri[2] - p
    π° = tri[3] - p
    π»β = π? Γ π―
    π»β = π― Γ π°
    dββ = π»β β π»β
    # Test the normals point the same direction relative to each other
    # and that surface normals are equivalent using π»Μ β π»Μ = 1
    # dββ > 0 is redundant if the point is in the triangle, but it is a very 
    # fast check that the point is in the plane of the triangle.
    (dββ > 0 && dββ β norm(π»β) * norm(π»β)) || return false
    # We need only check the direction of the norm of the last triangle to 
    # prove that the point is in the triangle
    return π»β β (π° Γ π?) > 0
end
