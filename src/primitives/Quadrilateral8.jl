# A quadrilateral with quadratic edges.
struct Quadrilateral8{N,T} <: Face{N,T}
    # The points are assumed to be ordered  in counter clockwise order as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = vertex D
    # p₅ = point on the quadratic segment from A to B
    # p₆ = point on the quadratic segment from B to C
    # p₇ = point on the quadratic segment from C to D
    # p₈ = point on the quadratic segment from D to A
    points::SVector{8, Point{N,T}}
end

const Quadrilateral8_2D = Quadrilateral8{2}
const Quadrilateral8_3D = Quadrilateral8{3}

Base.@propagate_inbounds function Base.getindex(q::Quadrilateral8, i::Integer)
    getfield(q, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function Quadrilateral8(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}, 
                        p₄::Point{N,T}, p₅::Point{N,T}, p₆::Point{N,T},
                        p₇::Point{N,T}, p₈::Point{N,T},) where {N,T}
    return Quadrilateral8{N,T}(SVector{8, Point{N,T}}(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
end
function Quadrilateral8{N}(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}, 
                           p₄::Point{N,T}, p₅::Point{N,T}, p₆::Point{N,T},
                           p₇::Point{N,T}, p₈::Point{N,T},) where {N,T}
    return Quadrilateral8{N,T}(SVector{8, Point{N,T}}(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
end

# Methods
# ---------------------------------------------------------------------------------------------
function (quad8::Quadrilateral8)(r, s)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/4)*quad8[1] +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/4)*quad8[2] +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/4)*quad8[3] +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/4)*quad8[4] +
                            ((1 - ξ^2)*(1 - η)/2)*quad8[5] +
                            ((1 - η^2)*(1 + ξ)/2)*quad8[6] +
                            ((1 - ξ^2)*(1 + η)/2)*quad8[7] +
                            ((1 - η^2)*(1 - ξ)/2)*quad8[8] )
end

function (quad8::Quadrilateral8)(p::Point_2D)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r = p[1]; s = p[2]
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/4)*quad8[1] +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/4)*quad8[2] +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/4)*quad8[3] +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/4)*quad8[4] +
                            ((1 - ξ^2)*(1 - η)/2)*quad8[5] +
                            ((1 - η^2)*(1 + ξ)/2)*quad8[6] +
                            ((1 - ξ^2)*(1 + η)/2)*quad8[7] +
                            ((1 - η^2)*(1 - ξ)/2)*quad8[8] )
end

# This can surely be turned into an algebraic equation using Mathematica
# to solve the double integral over the determinant of the jacobian
area(quad8::Quadrilateral8_2D) = area(quad8, Val(2))
function area(quad8::Quadrilateral8{N,T}, ::Val{P}) where {N,T,P}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #                           1  1
    # A = ∬ ‖∂Q/∂r × ∂Q/∂s‖dA = ∫  ∫ ‖∂Q/∂r × ∂Q/∂s‖ ds dr
    #      D                    0  0
    #
    #       P   P
    #   =   ∑   ∑  wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #      i=1 j=1
    w, r = gauss_legendre_quadrature(T, Val(P))
    a = T(0)
    for j = 1:P, i = 1:P
        J = 𝗝(quad8, r[i], r[j]) 
        a += w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2))
    end
    return a
end

centroid(quad8::Quadrilateral8_2D) = centroid(quad8, Val(3))
function centroid(quad8::Quadrilateral8_2D{T}, ::Val{P}) where {P,T}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #                           1  1
    # A = ∬ ‖∂Q/∂r × ∂Q/∂s‖dA = ∫  ∫ ‖∂Q/∂r × ∂Q/∂s‖ ds dr
    #      D                    0  0
    #
    #       P   P
    #   =   ∑   ∑  wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #      i=1 j=1
    #
    # C_x = (∫∫ x dA)/A, C_y = (∫∫ y dA)/A
    #         D                  D
    w, r = gauss_legendre_quadrature(T, Val(P))
    A = T(0)
    C = Point_2D{T}(0,0)
    for j = 1:P, i = 1:P
        J = 𝗝(quad8, r[i], r[j]) 
        weighted_val = w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2))
        A += weighted_val 
        C += weighted_val * quad8(r[i], r[j]) 
    end
    return C/A
end

function jacobian(quad8::Quadrilateral8, r, s)
    # Chain rule
    # ∂Q   ∂Q ∂ξ     ∂Q      ∂Q   ∂Q ∂η     ∂Q
    # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
    # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
    ξ = 2r - 1; η = 2s - 1
    ∂Q_∂ξ = (1 - η)*(2ξ + η)/4*quad8[1] +
            (1 - η)*(2ξ - η)/4*quad8[2] +
            (1 + η)*(2ξ + η)/4*quad8[3] +
            (1 + η)*(2ξ - η)/4*quad8[4] +
                    -ξ*(1 - η)*quad8[5] +
                   (1 - η^2)/2*quad8[6] +
                    -ξ*(1 + η)*quad8[7] +
                  -(1 - η^2)/2*quad8[8]

    ∂Q_∂η = (1 - ξ)*( ξ + 2η)/4*quad8[1] +
            (1 + ξ)*(-ξ + 2η)/4*quad8[2] +
            (1 + ξ)*( ξ + 2η)/4*quad8[3] +
            (1 - ξ)*(-ξ + 2η)/4*quad8[4] +
                   -(1 - ξ^2)/2*quad8[5] +
                     -η*(1 + ξ)*quad8[6] +
                    (1 - ξ^2)/2*quad8[7] +
                     -η*(1 - ξ)*quad8[8]

    return hcat(2*∂Q_∂ξ, 2*∂Q_∂η)
end

function Base.in(p::Point_2D, quad8::Quadrilateral8_2D)
    # If the point is to the left of every edge
    #  4<-----3
    #  |      ^
    #  | p    |
    #  |      |
    #  |      |
    #  v----->2
    #  1
    return isleft(p, QuadraticSegment_2D(quad8[1], quad8[2], quad8[5])) &&
           isleft(p, QuadraticSegment_2D(quad8[3], quad8[4], quad8[7])) &&
           isleft(p, QuadraticSegment_2D(quad8[2], quad8[3], quad8[6])) &&
           isleft(p, QuadraticSegment_2D(quad8[4], quad8[1], quad8[8]))
end

# function in(p::Point_2D{F}, quad8::Quadrilateral8_2D{F}, N::Int64) where {F <: AbstractFloat}
#     # Determine if the point is in the triangle using the Newton-Raphson method
#     # N is the max number of iterations of the method.
#     p_rs = real_to_parametric(p, quad8, N)
#     ϵ = parametric_coordinate_ϵ
#     if (-ϵ ≤ p_rs[1] ≤ 1 + ϵ) &&
#        (-ϵ ≤ p_rs[2] ≤ 1 + ϵ)
#         return true
#     else
#         return false
#     end
# end

function Base.intersect(l::LineSegment_2D{T}, quad8::Quadrilateral8_2D{T}) where {T} 
    # Create the 3 line segments that make up the triangle and intersect each one
    p₁ = Point_2D{T}(0,0)
    p₂ = Point_2D{T}(0,0)
    p₃ = Point_2D{T}(0,0)
    p₄ = Point_2D{T}(0,0)
    p₅ = Point_2D{T}(0,0)
    p₆ = Point_2D{T}(0,0)
    npoints = 0x0000
    for i ∈ 1:4
        hits, points = l ∩ QuadraticSegment_2D(quad8[(i - 1) % 4 + 1], 
                                               quad8[      i % 4 + 1], 
                                               quad8[          i + 4]) 
        for point in view(points, 1:Int64(hits))
            npoints += 0x0001
            if npoints === 0x0001
                p₁ = point
            elseif npoints === 0x0002
                p₂ = point
            elseif npoints === 0x0003
                p₃ = point
            elseif npoints === 0x0004
                p₄ = point
            elseif npoints === 0x0005
                p₅ = point
            else
                p₆ = point
            end
        end
    end 
    return npoints, SVector(p₁, p₂, p₃, p₄, p₅, p₆) 
end

# function intersect(l::LineSegment_2D, quad8::Quadrilateral8_2D)
#     # Create the 3 quadratic segments that make up the triangle and intersect each one
#     edges = SVector(QuadraticSegment_2D(quad8[1], quad8[2], quad8[5]),
#                     QuadraticSegment_2D(quad8[2], quad8[3], quad8[6]),
#                     QuadraticSegment_2D(quad8[3], quad8[4], quad8[7]),
#                     QuadraticSegment_2D(quad8[4], quad8[1], quad8[8]))
#     ipoints = MVector(Point_2D(), Point_2D(), Point_2D(),
#                       Point_2D(), Point_2D(), Point_2D())
#     n_ipoints = 0x00000000
#     # We need to account for 6 points returned
#     for k ∈ 1:4
#         npoints, points = l ∩ edges[k]
#         for i = 1:npoints
#             n_ipoints += 0x00000001
#             ipoints[n_ipoints] = points[i]
#         end
#     end
#     return n_ipoints, SVector(ipoints)
# end
# 
function real_to_parametric(p::Point_2D, quad8::Quadrilateral8_2D)
    return real_to_parametric(p, quad8, 30)
end

function real_to_parametric(p::Point_2D{T}, quad8::Quadrilateral8_2D{T}, 
                            max_iters::Int64) where {T}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method.
    # If a conversion doesn't exist, the minimizer is returned.
    # Initial guess at centroid
    rs = SVector{2,T}(1//2, 1//2) + inv(𝗝(quad8, 1//2, 1//2))*(p - quad8(1//3, 1//2))
    for i ∈ 1:max_iters
        Δrs = inv(𝗝(quad8, rs[1], rs[2]))*(p - quad8(rs[1], rs[2])) 
        if Δrs ⋅ Δrs < T((1e-6)^2)
            break
        end
        rs += Δrs 
    end
    return Point_2D{T}(rs[1], rs[2])
end

function triangulate(quad8::Quadrilateral8{N,T}, D::Int64) where {N,T}
    # D is the number of divisions of each edge
    triangles = Vector{Triangle{N,T}}(undef, 2*(D+1)*(D+1))
    if N === 0
        triangles[1] = Triangle(quad8[1], quad8[2], quad8[3])
        triangles[2] = Triangle(quad8[3], quad8[4], quad8[1])
    else
        for j = 0:D, i = 0:D
            triangles[2*(D+1)*j + 2i + 1] = Triangle(quad8(    i/(D+1),     j/(D+1)),
                                                     quad8((i+1)/(D+1),     j/(D+1)),
                                                     quad8(    i/(D+1), (j+1)/(D+1)))
            triangles[2*(D+1)*j + 2i + 2] = Triangle(quad8(    i/(D+1), (j+1)/(D+1)),
                                                     quad8((i+1)/(D+1),     j/(D+1)),
                                                     quad8((i+1)/(D+1), (j+1)/(D+1)))
        end
    end
    return triangles
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, quad8::Quadrilateral8)
        q₁ = QuadraticSegment(quad8[1], quad8[2], quad8[5])
        q₂ = QuadraticSegment(quad8[2], quad8[3], quad8[6])
        q₃ = QuadraticSegment(quad8[3], quad8[4], quad8[7])
        q₄ = QuadraticSegment(quad8[4], quad8[1], quad8[8])
        qsegs = [q₁, q₂, q₃, q₄]
        return convert_arguments(LS, qsegs)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:Quadrilateral8})
        point_sets = [convert_arguments(LS, quad8) for quad8 ∈ Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, quad8::Quadrilateral8)
        triangles = triangulate(quad8, 13)
        return convert_arguments(M, triangles)
    end

    function convert_arguments(M::Type{<:Mesh}, Q::Vector{<:Quadrilateral8})
        triangles = reduce(vcat, triangulate.(Q, 13))
        return convert_arguments(M, triangles)
    end
end
