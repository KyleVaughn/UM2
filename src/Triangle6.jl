import Base: intersect
using StaticArrays

# NOTE: The N (number of edge subdivisions) used in triangulation for the intersection
# algorithm and the ∈  algorithm must be the same.

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
function (tri6::Triangle6)(r::T, s::T) where {T <: AbstractFloat}
    return (1 - r - s)*(2(1 - r - s) - 1)*tri6.points[1] +
                                 r*(2r-1)*tri6.points[2] +
                                 s*(2s-1)*tri6.points[3] +
                           4r*(1 - r - s)*tri6.points[4] +
                                   (4r*s)*tri6.points[5] +
                           4s*(1 - r - s)*tri6.points[6]
end

function derivatives(tri6::Triangle6{T}, r::T, s::T) where {T <: AbstractFloat}
    # Return ( dtri6/dr(r, s), dtri6/ds(r, s) )
    d_dr = (4r + 4s - 3)*tri6.points[1] + 
                (4r - 1)*tri6.points[2] +
           4(1 - 2r - s)*tri6.points[4] +     
                    (4s)*tri6.points[5] +
                   (-4s)*tri6.points[6]

    d_ds = (4r + 4s - 3)*tri6.points[1] + 
                (4s - 1)*tri6.points[3] +
                   (-4r)*tri6.points[4] +
                    (4r)*tri6.points[5] +
           4(1 - r - 2s)*tri6.points[6]     
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
    w, rs = gauss_legendre_quadrature(tri6, N)
    return sum(w .* norm.([dr × ds for (dr, ds) in [derivatives(tri6, r, s) for (r, s) in rs]]))
end

function triangulate(tri6::Triangle6{T}, N::Int64) where {T <: AbstractFloat}
    triangles = Vector{Triangle{T}}(undef, N*N + 2N + 1)
    if N == 0
        triangles[1] = Triangle(tri6.points[1], tri6.points[2], tri6.points[3])
    else
        i = 1
        for S = 0:N, R = 0:N-S
            triangles[i] = Triangle(tri6(T(    R/(N+1)), T(    S/(N+1))), 
                                    tri6(T((R+1)/(N+1)), T(    S/(N+1))), 
                                    tri6(T(    R/(N+1)), T((S+1)/(N+1))))
            i += 1
        end
        j = 1 + ((N+1)*(N+2)) ÷ 2
        for S = 1:N, R = 0:N-S
            triangles[j] = Triangle(tri6(T(    R/(N+1)), T(    S/(N+1))), 
                                    tri6(T((R+1)/(N+1)), T((S-1)/(N+1))), 
                                    tri6(T((R+1)/(N+1)), T(    S/(N+1))))
            j += 1
        end
    end
    return triangles 
end

function intersect(l::LineSegment{T}, tri6::Triangle6{T}; N::Int64 = 12) where {T <: AbstractFloat}
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
        return true, 2, ipoints
    else
        return false, -1, ipoints
    end
end

function in(p::Point{T}, tri6::Triangle6{T}; N::Int64 = 12) where {T <: AbstractFloat}
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
