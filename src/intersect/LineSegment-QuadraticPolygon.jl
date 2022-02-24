function Base.intersect(l::LineSegment2D{T}, poly::QuadraticPolygon{N, 2, T}
                       ) where {N, T <:Union{Float32, Float64}} 
    # Create the quadratic segments that make up the polygon and intersect each one
    points = zeros(MVector{N, Point2D{T}})
    npoints = 0x0000
    M = N ÷ 2
    for i ∈ 1:M-1
        hits, ipoints = l ∩ QuadraticSegment2D(poly[i], poly[i + 1], poly[i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
    end
    hits, ipoints = l ∩ QuadraticSegment2D(poly[M], poly[1], poly[N])
    for j in 1:hits
        npoints += 0x0001
        points[npoints] = ipoints[j]
    end
    return npoints, SVector(points)
end

# Cannot mutate BigFloats in an MVector, so we use a regular Vector
function Base.intersect(l::LineSegment2D{BigFloat}, poly::QuadraticPolygon{N, 2, BigFloat}
                       ) where {N} 
    # Create the quadratic segments that make up the polygon and intersect each one
    points = zeros(Point2D{BigFloat}, N)
    npoints = 0x0000
    M = N ÷ 2
    for i ∈ 1:M-1
        hits, ipoints = l ∩ QuadraticSegment2D(poly[i], poly[i + 1], poly[i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
    end
    hits, ipoints = l ∩ QuadraticSegment2D(poly[M], poly[1], poly[N])
    for j in 1:hits
        npoints += 0x0001
        points[npoints] = ipoints[j]
    end
    return npoints, SVector{N, Point2D{BigFloat}}(points)
end

# A quadratic triangle, defined in 3D.

# Summary of intersection methods:
#       Triangulation
#           - Speed varies dramatically with precision
#               - The faster method when precision at or below 2 decimal places is desired
#           - Can provide false positives and false negatives due to approximation of the surface
#           - Constant time for a given number of triangles
#               - The only way to determine if two intersections exist is to test until two unique
#                 points are found. Since this happens infrequently, all points are tested, so time
#                 isn't wasted on control logic.
#       Newton-Raphson (iterative)
#           - Speed varies slightly based upon line segment length and orientation
#               - A shorter line segment will converge faster
#               - Converges based upon Jacobian matrix.
#                   - If derivatives are small, the iteration can become slow
#               - The faster method when precision beyond 2 decimal places is desired
#           - Less accurate
#               - May falsely give one intersection instead of two, especially for longer segments.
#               - This is due to the point of convergence being dependent on the initial guess
#                 point. The two starting points are placed close to the line segment start/stop
#                 to try to mitigate this.
#           - Precision to 6+ decimal places.
#       Overall
#           - Triangulation is predictable in speed, slow for high precision, and largely accurate
#           - Newton-Raphson is unpredictable in speed, fast for high precision, and
#               can be inaccurate for 2 intersections
#           - Consider the difference in timing between intersections for porting to GPU
#               - Newton-Raphson may cause thread divergence


# Triangulate, then intersect
function intersect(l::LineSegment_3D{T}, tri6::Triangle6_3D{T};
        N::Int64 = 25) where {T <: AbstractFloat}
    triangles = triangulate(tri6, N)
    npoints = 0
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 0)
    intersections = l .∩ triangles
    bools = map(x->x[1], intersections)
    points = map(x->x[2], intersections)
    npoints = count(bools)
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 0)
    if npoints === 0
        return false, 0, p₁, p₂
    elseif npoints === 1
        p₁ = points[argmax(bools)]
        return true, 1, p₁, p₂
    elseif npoints === 2
        indices = findall(bools)
        p₁ = points[indices[1]]
        p₂ = points[indices[2]]
        # Check uniqueness
        if p₁ ≈ p₂
            return true, 1, p₁, p₂
        else
            return true, 2, p₁, p₂
        end
    else
        # Account for 3 points and 4 points?
        # If intersection is on edge shared by two triangles on entrance and/or exit 3/4 intersections
        # can be detected
        return true, -1, p₁, p₂
    end
end

# A more exact intersection algorithm that triangulation, uses Newton-Raphson.
function intersect_iterative(l::LineSegment_3D{T}, tri6::Triangle6_3D{T};
        N::Int64=30) where {T <: AbstractFloat}
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 0)
    npoints = 0
    u⃗ = l.points[2] - l.points[1]
    ray_start = real_to_parametric(l(0), tri6; N=10) # closest r,s to the ray start
    ray_stop  = real_to_parametric(l(1), tri6; N=10) # closest r,s to the ray stop
    # The parametric coordinates corresponding to the start of the line segment
    r₁ = ray_start[1]
    s₁ = ray_start[2]
    t₁ = T(0)
    # The parametric coordinates corresponding to the start of the line segment
    r₂ = ray_stop[1]
    s₂ = ray_stop[2]
    t₂ = T(1)
    # Iteration for point 1
    err₁ = tri6(r₁, s₁) - l(t₁)
    for i = 1:N
        ∂r₁, ∂s₁ = derivatives(tri6, r₁, s₁)
        J₁ = hcat(∂r₁.x, ∂s₁.x, -u⃗.x)
        # If matrix is singular, it's probably bad luck. Just perturb it a bit.
        if abs(det(J₁)) < 1e-5
            r₁, s₁, t₁ = [r₁, s₁, t₁] + rand(3)/10
        else
            r₁, s₁, t₁ = [r₁, s₁, t₁] - inv(J₁) * err₁.x
            errₙ = tri6(r₁, s₁) - l(t₁)
            if norm(errₙ - err₁) < 5e-6
                break
            end
            err₁ = errₙ
        end
    end
    # Iteration for point 2
    err₂ = tri6(r₂, s₂) - l(t₂)
    for j = 1:N
        ∂r₂, ∂s₂ = derivatives(tri6, r₂, s₂)
        J₂ = hcat(∂r₂.x, ∂s₂.x, -u⃗.x)
        if abs(det(J₂)) < 1e-5
            r₂, s₂, t₂ = [r₂, s₂, t₂] + rand(3)/10
        else
            r₂, s₂, t₂ = [r₂, s₂, t₂] - inv(J₂) * err₂.x
            errₘ = tri6(r₂, s₂) - l(t₂)
            if norm(errₘ - err₂) < 5e-6
                break
            end
            err₂ = errₘ
        end
    end

    p₁ = l(t₁)
    if (0 ≤ r₁ ≤ 1) && (0 ≤ s₁ ≤ 1) && (0 ≤ t₁ ≤ 1) && (p₁ ≈ tri6(r₁, s₁))
        npoints += 1
    end

    p₂ = l(t₂)
    if (0 ≤ r₂ ≤ 1) && (0 ≤ s₂ ≤ 1) && (0 ≤ t₂ ≤ 1) && (p₂ ≈ tri6(r₂, s₂))
        npoints += 1
        # If only point 2 is valid, return it as p₁
        # If points are duplicate, reduce npoints
        if npoints === 2 && p₁ ≈ p₂
            npoints -= 1
        elseif npoints === 1
            p₁ = p₂
        end
    end
    return npoints > 0, npoints, p₁, p₂
end

