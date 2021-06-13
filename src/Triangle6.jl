import Base: intersect
using StaticArrays

# NOTE: The N (number of edge subdivisions) used in triangulation for the intersection
# algorithm and the ∈  algorithm must be the same for consistent ray tracing.

# Most of the intersection and in algorithm compute time is the triangulation. How can we speed that up?

struct Triangle6{T <: AbstractFloat} <: Face
    points::NTuple{6, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6(p₁::Point{T}, 
         p₂::Point{T}, 
         p₃::Point{T},
         p₄::Point{T},
         p₅::Point{T},
         p₆::Point{T}
        ) where {T <: AbstractFloat} = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
function (tri6::Triangle6{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    r_T = T(r)
    s_T = T(s)
    return (1 - r_T - s_T)*(2(1 - r_T - s_T) - 1)*tri6.points[1] +
                                     r_T*(2r_T-1)*tri6.points[2] +
                                     s_T*(2s_T-1)*tri6.points[3] +
                             4r_T*(1 - r_T - s_T)*tri6.points[4] +
                                       (4r_T*s_T)*tri6.points[5] +
                             4s_T*(1 - r_T - s_T)*tri6.points[6]
end

function derivatives(tri6::Triangle6{T}, r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # Return ( ∂tri6/∂r, ∂tri6/∂s )
    r_T = T(r)
    s_T = T(s)
    d_dr = (4r_T + 4s_T - 3)*tri6.points[1] + 
                  (4r_T - 1)*tri6.points[2] +
           4(1 - 2r_T - s_T)*tri6.points[4] +     
                      (4s_T)*tri6.points[5] +
                     (-4s_T)*tri6.points[6]

    d_ds = (4r_T + 4s_T - 3)*tri6.points[1] + 
                  (4s_T - 1)*tri6.points[3] +
                     (-4r_T)*tri6.points[4] +
                      (4r_T)*tri6.points[5] +
           4(1 - r_T - 2s_T)*tri6.points[6]     
    return (d_dr, d_ds) 
end

function area(tri6::Triangle6{T}; N::Int64=12) where {T <: AbstractFloat}
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
    w, rs = gauss_legendre_quadrature(tri6, N)
    a = T(0)
    for i in 1:N
        (r, s) = rs[i]
        (dr, ds) = derivatives(tri6, r, s)
        a += w[i] * norm(dr × ds)
    end
    return a
end

function triangulate(tri6::Triangle6{T}, N::Int64) where {T <: AbstractFloat}
    triangles = Vector{Triangle{T}}(undef, (N+1)*(N+1))
    if N == 0
        triangles[1] = Triangle(tri6.points[1], tri6.points[2], tri6.points[3])
    else
        l_R = N + 2
        R = T.(LinRange(0, 1, l_R))
        m = 1
        for j = 1:l_R, i = 1:l_R-j
            triangles[m] = Triangle(tri6(R[i  ], R[j  ]), 
                                    tri6(R[i+1], R[j  ]), 
                                    tri6(R[i  ], R[j+1]))
            m += 1
        end
        n = 1 + ((N+1)*(N+2)) ÷ 2
        for j = 2:l_R, i = 1:l_R-j
            triangles[n] = Triangle(tri6(R[i  ], R[j  ]),
                                    tri6(R[i+1], R[j-1]), 
                                    tri6(R[i+1], R[j  ]))
            n += 1
        end
    end
    return triangles 
end

function intersect(l::LineSegment{T}, tri6::Triangle6{T}; N::Int64 = 13) where {T <: AbstractFloat}
    triangles = triangulate(tri6, N)
    intersections = l .∩ triangles
    bools = map(x->x[1], intersections)
    points = map(x->x[2], intersections)
    npoints = count(bools)
    ipoints = [points[1], points[1]]
    if npoints == 0
        return false, 0, ipoints
    elseif npoints == 1
        ipoints[1] = points[argmax(bools)]
        return true, 1, ipoints
    elseif npoints == 2
        indices = findall(bools)
        ipoints[1] = points[indices[1]]
        ipoints[2] = points[indices[2]]
        # Check uniqueness
        if ipoints[1] ≈ ipoints[2]
            return true, 1, ipoints
        else
            return true, 2, ipoints
        end
    else
        # Account for 3 points and 4 points?
        # If intersection is on edge shared by two triangles on entrance and/or exit 3/4 intersections
        # can be detected
        return true, -1, ipoints
    end
end

function in(p::Point{T}, tri6::Triangle6{T}; N::Int64 = 13) where {T <: AbstractFloat}
    return any(p .∈  triangulate(tri6, N))
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, tri6::Triangle6{T}) where {T <: AbstractFloat}
    q₁ = QuadraticSegment(tri6.points[1], tri6.points[2], tri6.points[4])
    q₂ = QuadraticSegment(tri6.points[2], tri6.points[3], tri6.points[5])
    q₃ = QuadraticSegment(tri6.points[3], tri6.points[1], tri6.points[6])
    qsegs = [q₁, q₂, q₃]
    return convert_arguments(P, qsegs)
end

function convert_arguments(P::Type{<:LineSegments}, 
        TA::AbstractArray{<:Triangle6{T}}) where {T <: AbstractFloat}
    point_sets = [convert_arguments(P, tri6) for tri6 in TA]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{Mesh{Tuple{Triangle6{T}}}}, 
        tri6::Triangle6{T}) where {T <: AbstractFloat}
    triangles = triangulate(tri6, 13)
    return convert_arguments(P, triangles)
end

function convert_arguments(MT::Type{Mesh{Tuple{Triangle6{T}}}},
        AT::Vector{Triangle{T}}) where {T <: AbstractFloat}
    points = reduce(vcat, [[tri.points[i].coord for i = 1:3] for tri in AT])
    faces = zeros(Int64, length(AT), 3)
    k = 1
    for i in 1:length(AT), j = 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(MT, points, faces)
end
