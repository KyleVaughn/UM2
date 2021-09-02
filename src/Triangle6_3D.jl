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

struct Triangle6_3D{T <: AbstractFloat}
    # The points are assumed to be ordered as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = point on the quadratic segment from A to B
    # p₅ = point on the quadratic segment from B to C
    # p₆ = point on the quadratic segment from C to A
    points::NTuple{6, Point_3D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6_3D(p₁::Point_3D{T}, 
         p₂::Point_3D{T}, 
         p₃::Point_3D{T},
         p₄::Point_3D{T},
         p₅::Point_3D{T},
         p₆::Point_3D{T}
        ) where {T <: AbstractFloat} = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
function (tri6::Triangle6_3D{T})(r::R, s::S) where {T <: AbstractFloat,
                                                    R <: Real,
                                                    S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T - s_T)*(2(1 - r_T - s_T) - 1)*tri6.points[1] +
                                     r_T*(2r_T-1)*tri6.points[2] +
                                     s_T*(2s_T-1)*tri6.points[3] +
                             4r_T*(1 - r_T - s_T)*tri6.points[4] +
                                       (4r_T*s_T)*tri6.points[5] +
                             4s_T*(1 - r_T - s_T)*tri6.points[6]
end

function (tri6::Triangle6_3D{T})(p::Point_2D{T}) where {T <: AbstractFloat, 
                                                        R <: Real,
                                                        S <: Real}
    r_T = p[1]
    s_T = p[2]
    return (1 - r_T - s_T)*(2(1 - r_T - s_T) - 1)*tri6.points[1] +
                                     r_T*(2r_T-1)*tri6.points[2] +
                                     s_T*(2s_T-1)*tri6.points[3] +
                             4r_T*(1 - r_T - s_T)*tri6.points[4] +
                                       (4r_T*s_T)*tri6.points[5] +
                             4s_T*(1 - r_T - s_T)*tri6.points[6]
end

function derivatives(tri6::Triangle6_3D{T}, r::R, s::S) where {T <: AbstractFloat, 
                                                               R <: Real,
                                                               S <: Real}
    # Let T(r,s) be the interpolation function for tri6
    # Returns ∂T/∂r, ∂T/∂s
    r_T = T(r)
    s_T = T(s)
    ∂T_∂r = (4r_T + 4s_T - 3)*tri6.points[1] + 
                   (4r_T - 1)*tri6.points[2] +
            4(1 - 2r_T - s_T)*tri6.points[4] +     
                       (4s_T)*tri6.points[5] +
                      (-4s_T)*tri6.points[6]

    ∂T_∂s = (4r_T + 4s_T - 3)*tri6.points[1] + 
                   (4s_T - 1)*tri6.points[3] +
                      (-4r_T)*tri6.points[4] +
                       (4r_T)*tri6.points[5] +
            4(1 - r_T - 2s_T)*tri6.points[6]     
    return ∂T_∂r, ∂T_∂s
end

function area(tri6::Triangle6_3D{T}; N::Int64=79) where {T <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
    # Let T(r,s) be the interpolation function for tri6,
    #                             1 1-r                          N
    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr = ∑ wᵢ||∂T/∂r(rᵢ,sᵢ) × ∂T/∂s(rᵢ,sᵢ)||
    #      D                      0  0                          i=1
    #
    # N is the number of points used in the quadrature.
    # See tuning/Triangle6_3D_area.jl for more info on how N = 79 was chosen.
    w, r, s = gauss_legendre_quadrature(tri6, N)
    a = T(0)
    for i in 1:N
        ∂T_∂r, ∂T_∂s = derivatives(tri6, r[i], s[i])
        a += w[i] * norm(∂T_∂r × ∂T_∂s)
    end
    return a
end

function triangulate(tri6::Triangle6_3D{T}, N::Int64) where {T <: AbstractFloat}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle_3D{T}}(undef, (N+1)*(N+1))
    if N === 0
        triangles[1] = Triangle_3D(tri6.points[1], tri6.points[2], tri6.points[3])
    else
        i = 1
        for S = 1:N, R = 0:N-S
            triangles[i]   = Triangle_3D(tri6(    R/(N+1),     S/(N+1)),
                                         tri6((R+1)/(N+1),     S/(N+1)),
                                         tri6(    R/(N+1), (S+1)/(N+1)))
            triangles[i+1] = Triangle_3D(tri6(    R/(N+1),     S/(N+1)),
                                         tri6((R+1)/(N+1), (S-1)/(N+1)),
                                         tri6((R+1)/(N+1),     S/(N+1)))
            i += 2
        end
        j = (N+1)*N + 1
        for S = 0:0, R = 0:N-S
            triangles[j] = Triangle_3D(tri6(    R/(N+1),     S/(N+1)),
                                       tri6((R+1)/(N+1),     S/(N+1)),
                                       tri6(    R/(N+1), (S+1)/(N+1)))
            j += 1
        end
    end 
    return triangles 
end

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

function real_to_parametric(p::Point_3D{T}, tri6::Triangle6_3D{T}; N::Int64=30) where {T <: AbstractFloat}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method. N is the max number of iterations
    # If a conversion doesn't exist, the minimizer is returned. 
    r = T(1//3) # Initial guess at triangle centroid
    s = T(1//3)
    err₁ = p - tri6(r, s)
    for i = 1:N
        ∂T_∂r, ∂T_∂s = derivatives(tri6, r, s)
        J = hcat(∂T_∂r.x, ∂T_∂s.x)
        Δr, Δs = J \ err₁.x 
        r = r + Δr
        s = s + Δs
        err₂ = p - tri6(r, s)
        if norm(err₂ - err₁) < 1e-6
            break
        end
        err₁ = err₂
    end
    return Point_2D(r, s)
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
