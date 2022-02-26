# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

(l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)

(q::QuadraticSegment)(r) = Point(((2r - 1)*( r - 1))q.𝘅₁ + 
                                 (( r    )*(2r - 1))q.𝘅₂ + 
                                 ((4r    )*( 1 - r))q.𝘅₃ )

(tri::Triangle)(r, s) = Point((1 - r - s)*tri[1] + 
                                        r*tri[2] + 
                                        s*tri[3] )

(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1] + 
                                    ((    r)*(1 - s))quad[2] +
                                    ((    r)*(    s))quad[3] + 
                                    ((1 - r)*(    s))quad[4] )

(tri6::QuadraticTriangle)(r, s) = Point(((2(1 - r - s) - 1)*(1 - r - s))tri6[1] +
                                        ((      r         )*(2r - 1   ))tri6[2] +
                                        ((      s         )*(2s - 1   ))tri6[3] +
                                        ((     4r         )*(1 - r - s))tri6[4] +
                                        ((     4r         )*(        s))tri6[5] +
                                        ((     4s         )*(1 - r - s))tri6[6] )

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

(tet::Tetrahedron)(r, s, t) = Point((1 - r - s - t)*tet[1] +
                                                  r*tet[2] + 
                                                  s*tet[3] + 
                                                  t*tet[4] )

(hex::Hexahedron)(r, s, t) = Point(((1 - r)*(1 - s)*(1 - t))hex[1] + 
                                   ((    r)*(1 - s)*(1 - t))hex[2] +  
                                   ((    r)*(    s)*(1 - t))hex[3] + 
                                   ((1 - r)*(    s)*(1 - t))hex[4] + 
                                   ((1 - r)*(1 - s)*(    t))hex[5] + 
                                   ((    r)*(1 - s)*(    t))hex[6] + 
                                   ((    r)*(    s)*(    t))hex[7] + 
                                   ((1 - r)*(    s)*(    t))hex[8] ) 

function (tet10::QuadraticTetrahedron)(r, s, t)
    u = 1 - r - s - t
    return Point(((2u-1)u)tet10[ 1] +
                 ((2r-1)r)tet10[ 2] +
                 ((2s-1)s)tet10[ 3] +
                 ((2t-1)t)tet10[ 4] +
                    (4u*r)tet10[ 5] +
                    (4r*s)tet10[ 6] +
                    (4s*u)tet10[ 7] +
                    (4u*t)tet10[ 8] +
                    (4r*t)tet10[ 9] +
                    (4s*t)tet10[10] )
end

function (hex20::QuadraticHexahedron)(r, s, t)
    ξ = 2r - 1; η = 2s - 1; ζ = 2t - 1
    return Point(((1 - ξ)*(1 - η)*(1 - ζ)*(-2 - ξ - η - ζ)/8)hex20[ 1] +
                 ((1 + ξ)*(1 - η)*(1 - ζ)*(-2 + ξ - η - ζ)/8)hex20[ 2] +
                 ((1 + ξ)*(1 + η)*(1 - ζ)*(-2 + ξ + η - ζ)/8)hex20[ 3] +
                 ((1 - ξ)*(1 + η)*(1 - ζ)*(-2 - ξ + η - ζ)/8)hex20[ 4] +
                 ((1 - ξ)*(1 - η)*(1 + ζ)*(-2 - ξ - η + ζ)/8)hex20[ 5] +
                 ((1 + ξ)*(1 - η)*(1 + ζ)*(-2 + ξ - η + ζ)/8)hex20[ 6] +
                 ((1 + ξ)*(1 + η)*(1 + ζ)*(-2 + ξ + η + ζ)/8)hex20[ 7] +
                 ((1 - ξ)*(1 + η)*(1 + ζ)*(-2 - ξ + η + ζ)/8)hex20[ 8] +
                            ((1 - ξ^2)*(1 - η  )*(1 - ζ  )/4)hex20[ 9] +
                            ((1 + ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[10] +
                            ((1 - ξ^2)*(1 + η  )*(1 - ζ  )/4)hex20[11] +
                            ((1 - ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[12] +
                            ((1 - ξ  )*(1 - η  )*(1 - ζ^2)/4)hex20[13] +
                            ((1 + ξ  )*(1 - η^2)*(1 - ζ^2)/4)hex20[14] +
                            ((1 + ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[15] +
                            ((1 - ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[16] +
                            ((1 - ξ^2)*(1 - η  )*(1 + ζ  )/4)hex20[17] +
                            ((1 + ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[18] +
                            ((1 - ξ^2)*(1 + η  )*(1 + ζ  )/4)hex20[19] +
                            ((1 - ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[20] )
