# NOTE: The N (number of edge subdivisions) used in triangulation for the intersection
# algorithm and the ∈  algorithm must be the same for consistent ray tracing.

# Most of the intersection and in algorithm compute time is the triangulation. How can we speed that up?

struct Triangle6_3D{T <: AbstractFloat}
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
function (tri6::Triangle6_3D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T - s_T)*(2(1 - r_T - s_T) - 1)*tri6.points[1] +
                                     r_T*(2r_T-1)*tri6.points[2] +
                                     s_T*(2s_T-1)*tri6.points[3] +
                             4r_T*(1 - r_T - s_T)*tri6.points[4] +
                                       (4r_T*s_T)*tri6.points[5] +
                             4s_T*(1 - r_T - s_T)*tri6.points[6]
end

function derivatives(tri6::Triangle6_3D{T}, r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # Return ( ∂tri6/∂r, ∂tri6/∂s )
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
    return (∂T_∂r, ∂T_∂s) 
end

function area(tri6::Triangle6_3D{T}; N::Int64=79) where {T <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
    # Let T(r,s) be the interpolation function for tri6,
    #                             1 1-r                          N
    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr = ∑ wᵢ||∂T/∂r(rᵢ,sᵢ) × ∂T/∂s(rᵢ,sᵢ)||
    #      D                      0  0                          i=1
    #
    # NOTE: for 2D, N = 12 appears to be sufficient. For 3D, N = 79 is preferred.
    # This is to ensure error in area less that about 1e-6. This was determined
    # experimentally, not mathematically, so more sophisticated analysis could be
    # performed. For really strange 3D shapes, we need greater than 79
    w, r, s = gauss_legendre_quadrature(tri6, N)
#    return mapreduce((w,r,s)->w*norm(×(derivatives(tri6, r, s))), +, w, r, s)
    a = T(0)
    for i in 1:N
        (∂T_∂r, ∂T_∂s) = derivatives(tri6, r[i], s[i])
        a += w[i] * norm(∂T_∂r × ∂T_∂s)
    end
    return a
end

function triangulate(tri6::Triangle6_3D{T}, N::Int64) where {T <: AbstractFloat}
    triangles = Vector{Triangle_3D{T}}(undef, (N+1)*(N+1))
    if N == 0
        triangles[1] = Triangle_3D(tri6.points[1], tri6.points[2], tri6.points[3])
    else
        i = 1
        for S = 1:N, R = 0:N-S
            triangles[i] = Triangle_3D(tri6(    R/(N+1),     S/(N+1)),
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

# Triangulate then intersect
function intersect(l::LineSegment_3D{T}, tri6::Triangle6_3D{T}; N::Int64 = 13) where {T <: AbstractFloat}
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
    if npoints == 0
        return false, 0, p₁, p₂
    elseif npoints == 1
        p₁ = points[argmax(bools)]
        return true, 1, p₁, p₂
    elseif npoints == 2
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

function global_to_parametric(p::Point_3D{T}, tri6::Triangle6_3D{T}; N::Int64=10) where {T <: AbstractFloat}
    r = T(1//4)
    s = T(1//4)
    # 10 iterations appears to be sufficient for all realistic use cases, even 100 units away.
    for i = 1:N
        err = p - tri6(r, s)
        ∂T_∂r, ∂T_∂s = derivatives(tri6, r, s)
        J = hcat(∂T_∂r.x, ∂T_∂s.x)
        Δr, Δs = J \ err.x 
        r = r + Δr
        s = s + Δs 
    end
    return Point_2D(r, s)
end

# A more exact intersection algorithm that's about 7 times slower using Newton-Raphson
function intersect_iterative(l::LineSegment_3D{T}, tri6::Triangle6_3D{T}) where {T <: AbstractFloat}
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 0)
    npoints = 0
    u⃗ = l.points[2] - l.points[1]
    ray_start = global_to_parametric(l(0), tri6; N=6) # closest r,s to the ray start
    ray_stop  = global_to_parametric(l(1), tri6; N=6) # closest r,s to the ray stop
    # The parametric coordinates corresponding to the start of the line segment
    r₁ = ray_start[1]
    s₁ = ray_start[2]
    t₁ = T(0)
    # The parametric coordinates corresponding to the start of the line segment
    r₂ = ray_stop[1]
    s₂ = ray_stop[2]
    t₂ = T(1)
    for i = 1:6
        err₁ = tri6(r₁, s₁) - l(t₁)
        ∂r₁, ∂s₁ = derivatives(tri6, r₁, s₁)
        J₁ = hcat(∂r₁.x, ∂s₁.x, -u⃗.x)
        # If matrix is singular, it's probably bad luck. Just perturb it a bit.
        if abs(det(J₁)) < 1e-5
            r₁, s₁, t₁ = [r₁, s₁, t₁] + rand(3)/10
        else
            r₁, s₁, t₁ = [r₁, s₁, t₁] - inv(J₁) * err₁.x
        end
        err₂ = tri6(r₂, s₂) - l(t₂)
        ∂r₂, ∂s₂ = derivatives(tri6, r₂, s₂)
        J₂ = hcat(∂r₂.x, ∂s₂.x, -u⃗.x)
        if abs(det(J₂)) < 1e-5
            r₂, s₂, t₂ = [r₂, s₂, t₂] + rand(3)/10
        else
            r₂, s₂, t₂ = [r₂, s₂, t₂] - inv(J₂) * err₂.x
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
        if npoints == 2 && p₁ ≈ p₂
            npoints -= 1
        elseif npoints == 1
            p₁ = p₂
        end
    end
    return npoints > 0, npoints, p₁, p₂
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, tri6::Triangle6_3D{T}) where {T <: AbstractFloat}
    q₁ = QuadraticSegment(tri6.points[1], tri6.points[2], tri6.points[4])
    q₂ = QuadraticSegment(tri6.points[2], tri6.points[3], tri6.points[5])
    q₃ = QuadraticSegment(tri6.points[3], tri6.points[1], tri6.points[6])
    qsegs = [q₁, q₂, q₃]
    return convert_arguments(P, qsegs)
end

function convert_arguments(P::Type{<:LineSegments}, 
        TA::AbstractArray{<:Triangle6_3D{T}}) where {T <: AbstractFloat}
    point_sets = [convert_arguments(P, tri6) for tri6 in TA]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{Mesh{Tuple{Triangle6_3D{T}}}}, 
        tri6::Triangle6_3D{T}) where {T <: AbstractFloat}
    triangles = triangulate(tri6, 13)
    return convert_arguments(P, triangles)
end

function convert_arguments(MT::Type{Mesh{Tuple{Triangle6_3D{T}}}},
        AT::Vector{Triangle_3D{T}}) where {T <: AbstractFloat}
    points = reduce(vcat, [[tri.points[i].x for i = 1:3] for tri in AT])
    faces = zeros(Int64, length(AT), 3)
    k = 1
    for i in 1:length(AT), j = 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(MT, points, faces)
end
