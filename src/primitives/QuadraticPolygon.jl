# A polygon with quadratic edges
#
# Example:          
# For a quadratic triangle the points are ordered as follows:
# p₁ = vertex A     
# p₂ = vertex B     
# p₃ = vertex C     
# p₄ = point on the quadratic segment from A to B
# p₅ = point on the quadratic segment from B to C
# p₆ = point on the quadratic segment from C to A
#
# Points are in counterclockwise order.
struct QuadraticPolygon{N, Dim, T} <:Face{Dim, 2, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
const QuadraticTriangle        = QuadraticPolygon{6}
const QuadraticQuadrilateral   = QuadraticPolygon{8}
const QuadraticTriangle2D      = QuadraticPolygon{6,2}
const QuadraticQuadrilateral2D = QuadraticPolygon{8,2}
const QuadraticTriangle3D      = QuadraticPolygon{6,3}
const QuadraticQuadrilateral3D = QuadraticPolygon{8,3}

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
# area
#
# In general:
# Let 𝗳(r,s) be a parameterization of surface S
# A = ∬ dS = ∬ ‖∂𝗳/∂r × ∂𝗳/∂s‖dr ds
#     S      T
function area(tri6::QuadraticTriangle2D)
    # Mathematica for this algebraic nightmare
    a = (tri6[6] - tri6[4]) × tri6[1].coord
    b = (tri6[4] - tri6[5]) × tri6[2].coord
    c = (tri6[5] - tri6[6]) × tri6[3].coord
    d = (tri6[1] - tri6[2]) × tri6[3].coord
    e = tri6[2]  × tri6[1]
    return (4(a + b + c) + d + e)/6
end

# This likely has a simple analytic solution that should be worked out
area(quad8::QuadraticQuadrilateral2D) = area(quad8, Val(2))
#Integrate[n, {r, 0, 1}, {s, 0, 1},
# Assumptions ->
#  Im[Subscript[a, x]] == 0 && Im[Subscript[a, y]] == 0 &&
#   Im[Subscript[b, x]] == 0 && Im[Subscript[b, y]] == 0 &&
#   Im[Subscript[c, x]] == 0 && Im[Subscript[c, y]] == 0 &&
#   Im[Subscript[d, x]] == 0 &&
#   Im[Subscript[d, y]] ==
#    0 &&
#                                                             \
#
#   Im[Subscript[e, x]] == 0 && Im[Subscript[e, y]] == 0 &&
#   Im[Subscript[f, x]] == 0 && Im[Subscript[f, y]] == 0 &&
#   Im[Subscript[g, x]] == 0 && Im[Subscript[g, y]] == 0 &&
#   Im[Subscript[h, x]] == 0 && Im[Subscript[h, y]] == 0]
# 
#   1/12 (4 Cy Dx - 4 Cx Dy + Ay Ex + Cy Ex + Dy Ex - Ax Ey - Cx Ey - 
#   Dx Ey + 7 Ay Fx + 35 Cy Fx + 7 Dy Fx - 12 Ey Fx - 7 Ax Fy - 
#   35 Cx Fy - 7 Dx Fy + 12 Ex Fy - 7 Ay Gx - 35 Cy Gx - 3 Dy Gx + 
#   12 Ey Gx + 28 Fy Gx + 7 Ax Gy + 35 Cx Gy + 3 Dx Gy - 12 Ex Gy - 
#   28 Fx Gy + By (4 Cx + Ex + 3 Fx - 7 Gx - Hx) - Ay Hx - Cy Hx - 
#   Dy Hx + 4 Ey Hx + 12 Fy Hx - 12 Gy Hx + Ax Hy + Cx Hy + Dx Hy - 
#   4 Ex Hy - 12 Fx Hy + 12 Gx Hy + Bx (-4 Cy - Ey - 3 Fy + 7 Gy + Hy))
function area(quad8::QuadraticQuadrilateral{Dim, T}, ::Val{P}) where {Dim, T, P}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #     1 1                          P   P
    # A = ∫ ∫ ‖∂Q/∂r × ∂Q/∂s‖ ds dr =  ∑   ∑  wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #     0 0                         i=1 j=1
    w, r = gauss_legendre_quadrature(T, Val(P))
    a = zero(T)
    for j ∈ 1:P, i ∈ 1:P 
        J = 𝗝(quad8, r[i], r[j]) 
        a += w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2)) 
    end 
    return a
end

centroid(quad8::QuadraticQuadrilateral2D) = centroid(quad8, Val(3))
function centroid(quad8::QuadraticQuadrilateral{Dim, T}, ::Val{N}) where {Dim, T, N}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #            1  1                        N   N               
    # A = ∬ dA = ∫  ∫ ‖∂Q/∂r × ∂Q/∂s‖ds dr = ∑   ∑ wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #     S      0  0                       j=1 i=1
    #                  1  N   N               
    # 𝗖 = (∬ 𝘅 dA)/A = -  ∑   ∑ 𝘅ᵢⱼwᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #      S           A j=1 i=1
    w, r = gauss_legendre_quadrature(T, Val(N))
    A = zero(T)
    𝗖 = @SVector zeros(T, Dim)
    for j ∈ 1:N, i ∈ 1:N
        J = 𝗝(quad8, r[i], r[j])
        weighted_val = w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2))
        𝗖 += weighted_val * quad8(r[i], r[j])
        A += weighted_val
    end
    return Point(𝗖)/A
end

centroid(tri6::QuadraticTriangle2D) = centroid(tri6, Val(6))
function centroid(tri6::QuadraticTriangle{Dim, T}, ::Val{N}) where {Dim, T, N} 
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #            1 1-r                       N                
    # A = ∬ dA = ∫  ∫ ‖∂F/∂r × ∂F/∂s‖ds dr = ∑ wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖
    #     S      0  0                       i=1
    #                  1  N                                 
    # 𝗖 = (∬ 𝘅 dA)/A = -  ∑ 𝘅 wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖ 
    #      S           A i=1
    w, r, s = gauss_legendre_quadrature(tri6, Val(N))
    A = zero(T)
    𝗖 = @SVector zeros(T, Dim)
    for i ∈ 1:N
        J = 𝗝(tri6, r[i], s[i])
        weighted_val = w[i] * norm(view(J, :, 1) × view(J, :, 2)) 
        𝗖 += weighted_val * tri6(r[i], s[i])
        A += weighted_val
    end
    return Point(𝗖)/A
end

function jacobian(quad8::QuadraticQuadrilateral, r, s)
    # Chain rule
    # ∂Q   ∂Q ∂ξ     ∂Q      ∂Q   ∂Q ∂η     ∂Q
    # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
    # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
    ξ = 2r - 1; η = 2s - 1
    ∂Q_∂ξ = ((1 - η)*(2ξ + η)/4)quad8[1] +
            ((1 - η)*(2ξ - η)/4)quad8[2] +
            ((1 + η)*(2ξ + η)/4)quad8[3] +
            ((1 + η)*(2ξ - η)/4)quad8[4] +
                    (-ξ*(1 - η))quad8[5] +
                   ((1 - η^2)/2)quad8[6] +
                    (-ξ*(1 + η))quad8[7] +
                  (-(1 - η^2)/2)quad8[8]

    ∂Q_∂η = ((1 - ξ)*( ξ + 2η)/4)quad8[1] +
            ((1 + ξ)*(-ξ + 2η)/4)quad8[2] +
            ((1 + ξ)*( ξ + 2η)/4)quad8[3] +
            ((1 - ξ)*(-ξ + 2η)/4)quad8[4] +
                   (-(1 - ξ^2)/2)quad8[5] +
                     (-η*(1 + ξ))quad8[6] +
                    ((1 - ξ^2)/2)quad8[7] +
                     (-η*(1 - ξ))quad8[8]

    return 2*hcat(∂Q_∂ξ, ∂Q_∂η)
end

function jacobian(tri6::QuadraticTriangle, r, s)
    # Let F(r,s) be the interpolation function for tri6
    ∂F_∂r = (4r + 4s - 3)tri6[1] +
                 (4r - 1)tri6[2] +
          (4(1 - 2r - s))tri6[4] +
                     (4s)tri6[5] +
                    (-4s)tri6[6]

    ∂F_∂s = (4r + 4s - 3)tri6[1] +
                 (4s - 1)tri6[3] +
                    (-4r)tri6[4] +
                     (4r)tri6[5] +
          (4(1 - r - 2s))tri6[6]
    return hcat(∂F_∂r, ∂F_∂s)
end

# Test if a 2D point is in a 2D quadratic polygon
function Base.in(p::Point2D, poly::QuadraticPolygon{N, 2, T}) where {N, T}
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

function Base.intersect(l::LineSegment2D{T}, poly::QuadraticPolygon{N, 2, T}
                       ) where {N, T <:Union{Float32, Float64}} 
    # Create the quadratic segments that make up the polygon and intersect each one
    points = zeros(MVector{N, Point2D{T}})
    npoints = 0x0000
    M = N ÷ 2
    for i ∈ 1:M
        hits, ipoints = l ∩ QuadraticSegment2D(poly[(i - 1) % M + 1],  
                                               poly[      i % M + 1],
                                               poly[          i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
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
    for i ∈ 1:M
        hits, ipoints = l ∩ QuadraticSegment2D(poly[(i - 1) % M + 1],  
                                               poly[      i % M + 1],
                                               poly[          i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
    end
    return npoints, SVector{N, Point2D{BigFloat}}(points)
end

function real_to_parametric(p::Point2D, poly::QuadraticPolygon{N, 2, T}) where {N, T} 
    return real_to_parametric(p, poly, 30)
end
# Convert from real coordinates to the polygon's local parametric coordinates using
# Newton-Raphson.
# If a conversion doesn't exist, the minimizer is returned.
# Initial guess at polygon centroid
function real_to_parametric(p::Point2D{T}, poly::QuadraticPolygon{N, 2, T}, 
                            max_iters::Int64) where {N, T}
    if N === 6 # Triangle
        rs = SVector{2,T}(1//3, 1//3)
    else # Quadrilateral
        rs = SVector{2,T}(1//2, 1//2)
    end
    for i ∈ 1:max_iters
        Δrs = inv(𝗝(poly, rs[1], rs[2]))*(p - poly(rs[1], rs[2]))
        if Δrs ⋅ Δrs < T((1e-8)^2)
            break
        end
        rs += Δrs
    end
    return Point2D{T}(rs[1], rs[2])
end

function triangulate(quad8::QuadraticQuadrilateral{Dim, T}, N::Int64) where {Dim, T}
    # N is the number of divisions of each edge
    N1 = N + 1
    triangles = Vector{Triangle{Dim, T}}(undef, 2N1^2)
    if N === 0
        triangles[1] = Triangle(quad8[1], quad8[2], quad8[3])
        triangles[2] = Triangle(quad8[3], quad8[4], quad8[1])
    else
        for j = 0:N
            s₀ = j/N1 
            s₁ = (j + 1)/N1
            for i = 0:N
                r₀ = i/N1 
                r₁ = (i + 1)/N1
                triangles[2N1*j + 2i + 1] = Triangle(quad8(r₀, s₀),
                                                     quad8(r₁, s₀),
                                                     quad8(r₀, s₁))
                triangles[2N1*j + 2i + 2] = Triangle(quad8(r₀, s₁),
                                                     quad8(r₁, s₀),
                                                     quad8(r₁, s₁))
            end
        end
    end
    return triangles
end

function triangulate(tri6::QuadraticTriangle{Dim, T}, N::Int64) where {Dim, T}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle{Dim, T}}(undef, (N + 1)^2)
    if N === 0
        triangles[1] = Triangle(tri6[1], tri6[2], tri6[3])
    else
        i = 1
        N1 = N + 1
        for s ∈ 1:N
            s₋₁ = (s-1)/N1
            s₀ = s/N1
            s₁ = (s + 1)/N1
            for r ∈ 0:N-s
                r₀ = r/N1
                r₁ = (r + 1)/N1
                triangles[i]   = Triangle(tri6(r₀, s₀), tri6(r₁, s₀ ), tri6(r₀, s₁))
                triangles[i+1] = Triangle(tri6(r₀, s₀), tri6(r₁, s₋₁), tri6(r₁, s₀))
                i += 2
            end
        end
        j = N1*N + 1
        s₀ = zero(T)
        s₁ = 1/N1
        for r ∈ 0:N
            r₀ = r/N1
            r₁ = (r + 1)/N1
            triangles[j] = Triangle(tri6(r₀, s₀), tri6(r₁, s₀), tri6(r₀, s₁))
            j += 1
        end
    end
    return triangles
end

# Interpolation
# ---------------------------------------------------------------------------------------------
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
function (tri6::QuadraticTriangle)(r, s)
    return Point(((1 - r - s)*(2(1 - r - s) - 1))tri6[1] +
                                     (r*(2r - 1))tri6[2] +
                                     (s*(2s - 1))tri6[3] +
                                 (4r*(1 - r - s))tri6[4] +
                                           (4r*s)tri6[5] +
                                 (4s*(1 - r - s))tri6[6] )
end

function (tri6::QuadraticTriangle)(p::Point2D)
    r = p[1]; s = p[2]
    return Point(((1 - r - s)*(2(1 - r - s) - 1))tri6[1] +
                                     (r*(2r - 1))tri6[2] +
                                     (s*(2s - 1))tri6[3] +
                                 (4r*(1 - r - s))tri6[4] +
                                           (4r*s)tri6[5] +
                                 (4s*(1 - r - s))tri6[6] )
end

function (quad8::QuadraticQuadrilateral)(r, s)
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/2)quad8[1] +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/2)quad8[2] +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/2)quad8[3] +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/2)quad8[4] +
                              ((1 - ξ^2)*(1 - η))quad8[5] +
                              ((1 - η^2)*(1 + ξ))quad8[6] +
                              ((1 - ξ^2)*(1 + η))quad8[7] +
                              ((1 - η^2)*(1 - ξ))quad8[8] ) / 2
end

function (quad8::QuadraticQuadrilateral)(p::Point2D)
    r = p[1]; s = p[2]
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/2)quad8[1] +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/2)quad8[2] +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/2)quad8[3] +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/2)quad8[4] +
                              ((1 - ξ^2)*(1 - η))quad8[5] +
                              ((1 - η^2)*(1 + ξ))quad8[6] +
                              ((1 - ξ^2)*(1 + η))quad8[7] +
                              ((1 - η^2)*(1 - ξ))quad8[8] ) / 2
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, poly::QuadraticPolygon{N}) where {N}
        M = N ÷ 2
        qsegs = [QuadraticSegment(poly[(i - 1) % M + 1],  
                                  poly[      i % M + 1],
                                  poly[          i + M]) for i = 1:M]
        return convert_arguments(LS, qsegs)
    end

    function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:QuadraticPolygon})
        point_sets = [convert_arguments(LS, poly) for poly ∈ P]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(P::Type{<:Mesh}, poly::QuadraticPolygon)
        triangles = triangulate(poly, 7)
        return convert_arguments(P, triangles)
    end

    function convert_arguments(M::Type{<:Mesh}, P::Vector{<:QuadraticPolygon})
        triangles = reduce(vcat, triangulate.(P, 7))
        return convert_arguments(M, triangles)
    end
end
