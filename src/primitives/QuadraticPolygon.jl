"""
    QuadraticPolygon(SVector{N, Point{Dim, T}})
    QuadraticPolygon(p₁::Point{Dim, T}}, p₂::Point{Dim, T}}, ...)

Construct a quadratic polygon with `N` counter-clockwise oriented vertices in 
`Dim`-dimensional space. Several aliases exist for convenience, e.g. 
QuadraticTriangle (`N`=6), QuadraticQuadrilateral (`N`=8), etc.

The ordering for vertices for a quadratic triangle is as follows:
p₁ = vertex A     
p₂ = vertex B     
p₃ = vertex C     
p₄ = point on the quadratic segment from A to B
p₅ = point on the quadratic segment from B to C
p₆ = point on the quadratic segment from C to A
"""
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

QuadraticPolygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T} = 
    QuadraticPolygon{N, Dim, T}(v)
QuadraticPolygon{N}(x...) where {N} = QuadraticPolygon(SVector(x))
QuadraticPolygon(x...) = QuadraticPolygon(SVector(x))

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
