import Base: intersect
using StaticArrays

# NOTE: The N (number of edge subdivisions) used in triangulation for the intersection
# algorithm and the ∈  algorithm must be the same for consistent ray tracing.

# Most of the intersection and in algorithm compute time is the triangulation. How can we speed that up?

struct Quadrilateral8{T <: AbstractFloat} <: Face
    points::NTuple{8, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral8(p₁::Point{T}, p₂::Point{T}, p₃::Point{T}, p₄::Point{T},
               p₅::Point{T}, p₆::Point{T}, p₇::Point{T}, p₈::Point{T}
        ) where {T <: AbstractFloat} = Quadrilateral8((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad8::Quadrilateral8)(r::T, s::T) where {T <: AbstractFloat}
    ξ = 2r - 1; η = 2s - 1
    return (1 - ξ)*(1 - η)*(-ξ - η - 1)/4*quad8.points[1] +
           (1 + ξ)*(1 - η)*( ξ - η - 1)/4*quad8.points[2] +
           (1 + ξ)*(1 + η)*( ξ + η - 1)/4*quad8.points[3] +
           (1 - ξ)*(1 + η)*(-ξ + η - 1)/4*quad8.points[4] +
                      (1 - ξ^2)*(1 - η)/2*quad8.points[5] +
                      (1 - η^2)*(1 + ξ)/2*quad8.points[6] +
                      (1 - ξ^2)*(1 + η)/2*quad8.points[7] +
                      (1 - η^2)*(1 - ξ)/2*quad8.points[8]
end

#function derivatives(quad8::Quadrilateral8{T}, r::T, s::T) where {T <: AbstractFloat}
#    # Return ( dquad8/dr(r, s), dquad8/ds(r, s) )
#    d_dr = (4r + 4s - 3)*quad8.points[1] + 
#                (4r - 1)*quad8.points[2] +
#           4(1 - 2r - s)*quad8.points[4] +     
#                    (4s)*quad8.points[5] +
#                   (-4s)*quad8.points[6]
#
#    d_ds = (4r + 4s - 3)*quad8.points[1] + 
#                (4s - 1)*quad8.points[3] +
#                   (-4r)*quad8.points[4] +
#                    (4r)*quad8.points[5] +
#           4(1 - r - 2s)*quad8.points[6]     
#    return (d_dr, d_ds) 
#end
#
#function area(quad8::Quadrilateral8{T}; N::Int64=12) where {T <: AbstractFloat}
#    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
#    # Let T(r,s) be the interpolation function for quad8,
#    #                             1 1-r                          N
#    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr = ∑ wᵢ||∂T/∂r(rᵢ,sᵢ) × ∂T/∂s(rᵢ,sᵢ)||
#    #      D                      0  0                          i=1
#    #
#    # NOTE: for 2D, N = 12 appears to be sufficient. For 3D, N = 79 is preferred.
#    w, rs = gauss_legendre_quadrature(quad8, N)
#    return sum(w .* norm.([dr × ds for (dr, ds) in [derivatives(quad8, r, s) for (r, s) in rs]]))
#end
#
#function triangulate(quad8::Quadrilateral8{T}, N::Int64) where {T <: AbstractFloat}
#    triangles = Vector{Triangle{T}}(undef, N*N + 2N + 1)
#    if N == 0
#        triangles[1] = Triangle(quad8.points[1], quad8.points[2], quad8.points[3])
#    else
#        i = 1
#        for S = 0:N, R = 0:N-S
#            triangles[i] = Triangle(quad8(T(    R/(N+1)), T(    S/(N+1))), 
#                                    quad8(T((R+1)/(N+1)), T(    S/(N+1))), 
#                                    quad8(T(    R/(N+1)), T((S+1)/(N+1))))
#            i += 1
#        end
#        j = 1 + ((N+1)*(N+2)) ÷ 2
#        for S = 1:N, R = 0:N-S
#            triangles[j] = Triangle(quad8(T(    R/(N+1)), T(    S/(N+1))), 
#                                    quad8(T((R+1)/(N+1)), T((S-1)/(N+1))), 
#                                    quad8(T((R+1)/(N+1)), T(    S/(N+1))))
#            j += 1
#        end
#    end
#    return triangles 
#end
#
#function intersect(l::LineSegment{T}, quad8::Quadrilateral8{T}; N::Int64 = 12) where {T <: AbstractFloat}
#    triangles = triangulate(quad8, N)
#    intersections = l .∩ triangles
#    bools = map(x->x[1], intersections)
#    points = map(x->x[2], intersections)
#    npoints = count(bools)
#    ipoints = [points[1], points[1]]
#    if npoints == 0
#        return false, 0, ipoints
#    elseif npoints == 1
#        ipoints[1] = points[argmax(bools)]
#        return true, 1, ipoints
#    elseif npoints == 2
#        indices = findall(bools)
#        ipoints[1] = points[indices[1]]
#        ipoints[2] = points[indices[2]]
#        return true, 2, ipoints
#    else
#        return false, -1, ipoints
#    end
#end
#
#function in(p::Point{T}, quad8::Quadrilateral8{T}; N::Int64 = 12) where {T <: AbstractFloat}
#    return any(p .∈  triangulate(quad8, N))
#end
#
## Plot
## -------------------------------------------------------------------------------------------------
#function convert_arguments(P::Type{<:LineSegments}, quad8::Quadrilateral8{T}) where {T <: AbstractFloat}
#    q₁ = QuadraticSegment(quad8.points[1], quad8.points[2], quad8.points[4])
#    q₂ = QuadraticSegment(quad8.points[2], quad8.points[3], quad8.points[5])
#    q₃ = QuadraticSegment(quad8.points[3], quad8.points[1], quad8.points[6])
#    qsegs = [q₁, q₂, q₃]
#    return convert_arguments(P, qsegs)
#end
#
#function convert_arguments(P::Type{<:LineSegments}, 
#        TA::AbstractArray{<:Quadrilateral8{T}}) where {T <: AbstractFloat}
#    point_sets = [convert_arguments(P, quad8) for quad8 in TA]
#    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
#end
#
#function convert_arguments(P::Type{Mesh{Tuple{Quadrilateral8{T}}}}, 
#        quad8::Quadrilateral8{T}) where {T <: AbstractFloat}
#    triangles = triangulate(quad8, 12)
#    return convert_arguments(P, triangles)
#end
#
#function convert_arguments(MT::Type{Mesh{Tuple{Quadrilateral8{T}}}},
#        AT::Vector{Triangle{T}}) where {T <: AbstractFloat}
#    points = reduce(vcat, [[tri.points[i].coord for i = 1:3] for tri in AT])
#    faces = zeros(Int64, length(AT), 3)
#    k = 1
#    for i in 1:length(AT), j = 1:3
#        faces[i, j] = k
#        k += 1
#    end
#    return convert_arguments(MT, points, faces)
#end
