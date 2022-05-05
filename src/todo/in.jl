# Is left 
# ---------------------------------------------------------------------------------------------
# If the point is left of the line segment in the 2D plane. 
#
# The segment's direction is from 𝘅₁ to 𝘅₂. Let 𝘂 = 𝘅₂ - 𝘅₁ and 𝘃 = 𝗽 - 𝘅₁ 
# We may determine if the angle θ between the point and segment is in [0, π] based on the 
# sign of 𝘂 × 𝘃, since 𝘂 × 𝘃 = ‖𝘂‖‖𝘃‖sin(θ). 
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# We allow points on the line (𝘂 × 𝘃 = 0) to be left, since this test is primarily 
# used to determine if a point is inside a polygon. A mesh is supposed to partition
# its domain, so if we do not allow points on the line, there will exist points in the 
# mesh which will not be in any face.
@inline function isleft(p::Point2D, l::LineSegment2D)
    return 0 ≤ l.𝘂 × (p - l.𝘅₁)
end

# Is left
# ---------------------------------------------------------------------------------------------
# If the point is left of the quadratic segment in the 2D plane.
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# If the segment is straight, we can perform the isleft check with the straight
# line from the  segment's start point to the segment's end point.
# If this condition isn't met, the segment's curvature must be accounted for.
# We find the point on the curve q that is nearest point to the point of interest.
# Call this point q_near. We then perform the isleft check with the tangent vector
# of q at q_near and the vector from q_near to p, p - q_near.
function isleft(p::Point2D, q::QuadraticSegment2D)
    if isstraight(q)
        𝘃₁ = q.𝘅₂ - q.𝘅₁
        𝘃₂ = p - q.𝘅₁
        return 𝘃₁ × 𝘃₂ > 0
    else
        # See nearest_point for an explanation of the math.
        # When the cubic has 3 real roots, the point must be inside the
        # curve of the segment. Meaning:
        #   If the segment curves left, the point is right.
        #   If the segment curves right, the point is left.
        # This way we save substantial time by bypassing the complex number arithmetic
        𝘂 = q.𝘂
        𝘃 = q.𝘃
        𝘄 = p - q.𝘅₁
        # f′(r) = ar³ + br² + cr + d = 0
        a = 4(𝘂 ⋅ 𝘂)
        b = 6(𝘂 ⋅ 𝘃)
        c = 2((𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄))
        d = -2(𝘃 ⋅ 𝘄)
        # Lagrange's method
        e₁ = s₀ = -b/a
        e₂ = c/a
        e₃ = -d/a
        A = 2e₁^3 - 9e₁*e₂ + 27e₃
        B = e₁^2 - 3e₂
        if A^2 - 4B^3 > 0 # one real root
            s₁ = ∛((A + √(A^2 - 4B^3))/2)
            if s₁ == 0
                s₂ = s₁
            else
                s₂ = B/s₁
            end
            r = (s₀ + s₁ + s₂)/3
            𝘃₁ = 𝗗(q, r)
            𝘃₂ = p - q(r)
            return 𝘃₁ × 𝘃₂ > 0
        else # three real roots
            return (q.𝘅₂ - q.𝘅₁) × (q.𝘅₃ - q.𝘅₁) < 0
        end
    end
end



# Nearest point
# ---------------------------------------------------------------------------------------------
# Find the point on 𝗾(r) closest to the point of interest 𝘆.
#
# Note: r ∈ [0, 1] is not necessarily true for this function, since it finds the minimizer
# of the function 𝗾(r), ∀r ∈ ℝ
# Find r which minimizes ‖𝘆 - 𝗾(r)‖, where 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁.
# This r also minimizes ‖𝘆 - 𝗾(r)‖²
# It can be shown that this is equivalent to finding the minimum of the quartic function
# ‖𝘆 - 𝗾(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
# The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
# 𝘄 = 𝘆 - 𝘅₁, a = 4(𝘂 ⋅ 𝘂), b = 6(𝘂 ⋅ 𝘃), c = 2[(𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄)], d = -2(𝘃 ⋅ 𝘄)
# Lagrange's method (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
# is used to find the roots.
function nearest_point(p::Point{Dim,T}, q::QuadraticSegment) where {Dim,T}
    𝘂 = q.𝘂
    𝘃 = q.𝘃
    𝘄 = p - q.𝘅₁
    # f′(r) = ar³ + br² + cr + d = 0
    a = 4(𝘂 ⋅ 𝘂)
    b = 6(𝘂 ⋅ 𝘃)
    c = 2((𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄))
    d = -2(𝘃 ⋅ 𝘄)
    # Lagrange's method
    e₁ = s₀ = -b/a
    e₂ = c/a
    e₃ = -d/a
    A = 2e₁^3 - 9e₁*e₂ + 27e₃
    B = e₁^2 - 3e₂
    if A^2 - 4B^3 > 0 # one real root
        s₁ = ∛((A + √(A^2 - 4B^3))/2)
        if s₁ == 0
            s₂ = s₁
        else
            s₂ = B/s₁
        end
        r = (s₀ + s₁ + s₂)/3
        return r, q(r)
    else # three real roots
        # Complex cube root
        t₁ = exp(log((A + √(complex(A^2 - 4B^3)))/2)/3)
        if t₁ == 0
            t₂ = t₁
        else
            t₂ = B/t₁
        end
        ζ₁ = Complex{T}(-1/2, √3/2)
        ζ₂ = conj(ζ₁)
        dist_min = typemax(T)
        r_near = zero(T)
        p_near = nan(Point{Dim,T})
        # Use the real part of each root
        for rᵢ in (real((s₀ +    t₁ +    t₂)/3),
                   real((s₀ + ζ₂*t₁ + ζ₁*t₂)/3),
                   real((s₀ + ζ₁*t₁ + ζ₂*t₂)/3))
            pᵢ = q(rᵢ)
            dist = distance²(pᵢ, p)
            if dist < dist_min
                dist_min = dist
                r_near = rᵢ
                p_near = pᵢ
            end
        end
        return r_near, p_near
    end
end


# Hyperplane 
# ---------------------------------------------------------------------------------------------
Base.in(p::Point, plane::Hyperplane) = p.coord ⋅ plane.𝗻 ≈ plane.d
in_halfspace(p::Point, plane::Hyperplane) = p.coord ⋅ plane.𝗻 - plane.d ≥ 0

# Given a point p and line l that lie in the plane. Check that the point is left of the line
function isleft(p::Point3D, l::LineSegment3D, plane::Hyperplane3D)
    # Since p and l ∈ plane, l.𝘂 × (p - l.𝘅₁) must either by in the exact same direction
    # as plane.𝗻 or the exact opposite direction. If the direction is the same, the point
    # is left of the line.
    return 0 ≤ (l.𝘂 × (p - l.𝘅₁)) ⋅ plane.𝗻
end

# AABox 
# ---------------------------------------------------------------------------------------------
@inline Base.in(p::Point2D, aab::AABox2D) = aab.xmin ≤ p[1] ≤ aab.xmax && 
                                            aab.ymin ≤ p[2] ≤ aab.ymax
@inline Base.in(p::Point3D, aab::AABox3D) = aab.xmin ≤ p[1] ≤ aab.xmax && 
                                            aab.ymin ≤ p[2] ≤ aab.ymax &&
                                            aab.zmin ≤ p[3] ≤ aab.zmax
# Point inside polygon
# ---------------------------------------------------------------------------------------------
# Test if a point is in a polygon for 2D points/polygons
function Base.in(p::Point2D, poly::Polygon{N, 2}) where {N}
    # Test if the point is to the left of each edge. 
    for i ∈ 1:N-1
        isleft(p, LineSegment2D(poly[i], poly[i + 1])) || return false
    end
    return isleft(p, LineSegment2D(poly[N], poly[1]))
end
# Not necessarily planar
#function Base.in(p::Point3D, poly::Polygon{N, 3}) where {N}
#    # Check if the point is even in the same plane as the polygon
#    plane = Hyperplane(poly[1], poly[2], poly[3])
#    p ∈ plane || return false
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
    # P ∈ ABC iff the surface normals of CCW triangles PAB, PBC, & PCA are equal.
    𝗮 = tri[1] - p
    𝗯 = tri[2] - p
    𝗰 = tri[3] - p
    𝗻₁= 𝗮 × 𝗯 
    𝗻₂= 𝗯 × 𝗰
    d₁₂ = 𝗻₁ ⋅ 𝗻₂
    # Test the normals point the same direction relative to each other
    # and that surface normals are equivalent using 𝗻̂ ⋅ 𝗻̂ = 1
    # d₁₂ > 0 is redundant if the point is in the triangle, but it is a very 
    # fast check that the point is in the plane of the triangle.
    (d₁₂ > 0 && d₁₂ ≈ norm(𝗻₁)*norm(𝗻₂)) || return false
    # We need only check the direction of the norm of the last triangle to 
    # prove that the point is in the triangle
    return 𝗻₂ ⋅(𝗰 × 𝗮) > 0 
end

# This performs much better than the default routine, which is logically equivalent.
# Better simd this way? Chaining isleft doesn't have the same performance improvement for
# triangles.
function Base.in(p::Point2D, quad::Quadrilateral2D)
    return isleft(p, LineSegment2D(quad[1], quad[2])) &&
           isleft(p, LineSegment2D(quad[2], quad[3])) &&
           isleft(p, LineSegment2D(quad[3], quad[4])) &&
           isleft(p, LineSegment2D(quad[4], quad[1]))
end

# Test if a 2D point is in a 2D quadratic polygon
function Base.in(p::Point2D, poly::QuadraticPolygon{N, 2, T}) where {N, T}
    # Test if the point is to the left of each edge. 
    bool = true
    M = N ÷ 2
    for i ∈ 1:M-1
        if !isleft(p, QuadraticSegment2D(poly[i], poly[i + 1], poly[i + M]))
            return false
        end
    end
    return isleft(p, QuadraticSegment2D(poly[M], poly[1], poly[N]))
end
