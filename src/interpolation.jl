# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

(l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)

(q::QuadraticSegment)(r) = Point(((2r-1)*(r-1))q.𝘅₁ + (r*(2r-1))q.𝘅₂ + (4r*(1-r))q.𝘅₃)

(tri::Triangle)(r, s) = Point((1 - r - s)*tri[1] + r*tri[2] + s*tri[3])

(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1] + (r*(1 - s))quad[2] +
                                                (r*s)quad[3] + ((1 - r)*s)quad[4])

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
