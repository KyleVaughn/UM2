# A quadratic polygon defined by the linear shape's vertices, followed 
# by an additional non-vertex point on each edge. 
# Points are in counterclockwise order.
# Example:          
# For a quadratic triangle the points are ordered as follows
# p₁ = vertex A     
# p₂ = vertex B     
# p₃ = vertex C     
# p₄ = point on the quadratic segment from A to B
# p₅ = point on the quadratic segment from B to C
# p₆ = point on the quadratic segment from C to A

struct QuadraticPolygon{N, Dim, T} <:Face{Dim, 2, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
const QuadraticTriangle        = QuadraticPolygon{6}
const QuadraticQuadrilateral   = QuadraticPolygon{8}
# When the time comes for 3D, use metaprogramming/eval to export 2D/3D consts
const QuadraticTriangle2D      = QuadraticPolygon{6,2}
const QuadraticQuadrilateral2D = QuadraticPolygon{8,2}

Base.@propagate_inbounds function Base.getindex(poly::QuadraticPolygon, i::Integer)
    getfield(poly, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function QuadraticPolygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T}
    return QuadraticPolygon{N, Dim, T}(v)
end
QuadraticPolygon{N}(x...) where {N} = QuadraticPolygon(SVector(x))
QuadraticPolygon(x...) = QuadraticPolygon(SVector(x))

# Methods
# ---------------------------------------------------------------------------------------------
function area(tri6::QuadraticTriangle2D)
    # Let F(r,s) be the interpolation function for tri6, and 𝗝(r,s) be the Jacobian of F 
    # at (r,s). |𝗝| is the Jacobian determinant 
    #                    1 1-r                N
    # A = ∬ |𝗝(r,s)|dA = ∫  ∫ |𝗝(r,s)|ds dr = ∑ wᵢ|𝗝(rᵢ,sᵢ)|
    #     D              0  0                i=1
    # Mathematica for this algebraic nightmare
#    a = tri6[1].x; g = tri6[1].y 
#    b = tri6[2].x; h = tri6[2].y
#    c = tri6[3].x; i = tri6[3].y
#    d = tri6[4].x; j = tri6[4].y
#    e = tri6[5].x; k = tri6[5].y
#    f = tri6[6].x; l = tri6[6].y
#    return (4g*(f - d) + 4h*(d - e) + 4i*(e - f) + 4a*(j - l) + 4b*(k - j) + 4c*(l - k)
#            + a*(i - h) + b*(g - i) + c*(h - g))/6
return (4(((tri6[6] - tri6[4]) × tri6[1].coord)  + 
          ((tri6[4] - tri6[5]) × tri6[2].coord)  +
          ((tri6[5] - tri6[6]) × tri6[3].coord)) +
          ((tri6[1] - tri6[2]) × tri6[3].coord)  +
                      tri6[2]  × tri6[1])/6
end
#area(quad8::QuadraticQuadrilateral2D) = area(quad8, Val(2))
# function area(quad8::QuadraticQuadrilateral{Dim, T}, ::Val{P}) where {N,T,P}
#     # Gauss-Legendre quadrature over a quadrilateral is used.
#     # Let Q(r,s) be the interpolation function for quad8,
#     #                           1  1
#     # A = ∬ ‖∂Q/∂r × ∂Q/∂s‖dA = ∫  ∫ ‖∂Q/∂r × ∂Q/∂s‖ ds dr
#     #      D                    0  0
#     #   
#     #       P   P
#     #   =   ∑   ∑  wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
#     #      i=1 j=1
#     w, r = gauss_legendre_quadrature(T, Val(P))
#     a = T(0)
#     for j = 1:P, i = 1:P 
#         J = 𝗝(quad8, r[i], r[j]) 
#         a += w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2)) 
#     end 
#     return a
# end

centroid(tri6::QuadraticTriangle2D) = centroid(tri6, Val(6))
function centroid(tri6::QuadraticTriangle, ::Val{N}) where {N} 
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #                    1 1-r                N                
    # A = ∬ |𝗝(r,s)|dA = ∫  ∫ |𝗝(r,s)|ds dr = ∑ wᵢ|𝗝(rᵢ,sᵢ)|
    #     D              0  0                i=1
    #   
    # C_x = (∫∫ x dA)/A, C_y = (∫∫ y dA)/A
    #         D                  D
    w, r, s = gauss_legendre_quadrature(tri6, Val(N))
    # We can reuse our computed weighted Jacobian determinants, since we need these
    # in the C_y, C_y, and A.
    weighted_vals = @. w * det(𝗝(tri6, r, s)) 
    points = tri6.(r, s)
    A = sum(weighted_vals)
    C_x = sum(getindex.(points, 1) .* weighted_vals)
    C_y = sum(getindex.(points, 2) .* weighted_vals)
    return Point2D(C_x/A, C_y/A)
end

function jacobian(tri6::QuadraticTriangle, r, s)
    # Let F(r,s) be the interpolation function for tri6
    ∂F_∂r = (4r + 4s - 3)*tri6[1] +
                 (4r - 1)*tri6[2] +
            4(1 - 2r - s)*tri6[4] +
                     (4s)*tri6[5] +
                    (-4s)*tri6[6]

    ∂F_∂s = (4r + 4s - 3)*tri6[1] +
                 (4s - 1)*tri6[3] +
                    (-4r)*tri6[4] +
                     (4r)*tri6[5] +
            4(1 - r - 2s)*tri6[6]
    return hcat(∂F_∂r, ∂F_∂s)
end

# Test if a point is in a polygon for 2D points/polygons
function Base.in(p::Point2D, poly::QuadraticPolygon{N,2,T}) where {N,T}
    # Test if the point is to the left of each edge. 
    bool = true
    M = N ÷ 2
    for i ∈ 1:M
        if !isleft(p, QuadraticSegment2D(poly[(i - 1) % M + 1], 
                                         poly[      i % M + 1],
                                         poly[          i + M]))
            bool = false
            break
        end
    end
    return bool
end
# 
# function Base.intersect(l::LineSegment2D{T}, poly::Polygon{N,2,T}
#                        ) where {N,T <:Union{Float32, Float64}} 
#     # Create the line segments that make up the triangle and intersect each one
#     points = zeros(MVector{N,Point2D{T}})
#     npoints = 0x0000
#     for i ∈ 1:N
#         hit, point = l ∩ LineSegment2D(poly[(i - 1) % N + 1], poly[i % N + 1]) 
#         if hit
#             npoints += 0x0001
#             @inbounds points[npoints] = point
#         end
#     end
#     return npoints, SVector(points)
# end
# 
# # Cannot mutate BigFloats in an MVector, so we use a regular Vector
# function Base.intersect(l::LineSegment2D{BigFloat}, poly::Polygon{N,2,BigFloat}) where {N} 
#     # Create the line segments that make up the triangle and intersect each one
#     points = zeros(Point2D{BigFloat}, N)
#     npoints = 0x0000
#     for i ∈ 1:N
#         hit, point = l ∩ LineSegment2D(poly[(i - 1) % N + 1], poly[i % N + 1]) 
#         if hit
#             npoints += 0x0001
#             @inbounds points[npoints] = point
#         end
#     end
#     return npoints, SVector{N,Point2D{BigFloat}}(points)
# end
# 
# Interpolation
# ---------------------------------------------------------------------------------------------
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
function (tri6::QuadraticTriangle)(r, s)
    return Point(((1 - r - s)*(2(1 - r - s) - 1))*tri6[1] +
                                     (r*(2r - 1))*tri6[2] +
                                     (s*(2s - 1))*tri6[3] +
                                 (4r*(1 - r - s))*tri6[4] +
                                           (4r*s)*tri6[5] +
                                 (4s*(1 - r - s))*tri6[6] )
end

function (tri6::QuadraticTriangle)(p::Point2D)
    r = p[1]; s = p[2]
    return Point(((1 - r - s)*(2(1 - r - s) - 1))*tri6[1] +
                                     (r*(2r - 1))*tri6[2] +
                                     (s*(2s - 1))*tri6[3] +
                                 (4r*(1 - r - s))*tri6[4] +
                                           (4r*s)*tri6[5] +
                                 (4s*(1 - r - s))*tri6[6] )
end

function (quad8::QuadraticQuadrilateral)(r, s)
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

function (quad8::QuadraticQuadrilateral)(p::Point2D)
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



# # Plot
# # ---------------------------------------------------------------------------------------------
# if enable_visualization
#     function convert_arguments(LS::Type{<:LineSegments}, poly::Polygon{N}) where {N}
#         lines = [LineSegment2D(poly[(i-1) % N + 1],
#                                poly[    i % N + 1]) for i = 1:N] 
#         return convert_arguments(LS, lines)
#     end
# 
#     function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:Polygon})
#         point_sets = [convert_arguments(LS, poly) for poly ∈  P]
#         return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
#     end
# 
#     # Need to implement triangulation before this can be done for a general polygon
#     # function convert_arguments(M::Type{<:Mesh}, poly::)
#     # end
#     #function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
#     #    points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
#     #    faces = zeros(Int64, length(T), 3)
#     #    k = 1
#     #    for i in 1:length(T), j = 1:3
#     #        faces[i, j] = k
#     #        k += 1
#     #    end
#     #    return convert_arguments(M, points, faces)
#     #end
# end
