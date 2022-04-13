# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

(l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)

(q::QuadraticSegment)(r) = Point(((2r - 1)*( r - 1))q.𝘅₁.coord + 
                                 (( r    )*(2r - 1))q.𝘅₂.coord + 
                                 ((4r    )*( 1 - r))q.𝘅₃.coord )

(tri::Triangle)(r, s) = Point((1 - r - s)*tri[1].coord + 
                                        r*tri[2].coord + 
                                        s*tri[3].coord )

(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1].coord + 
                                    ((    r)*(1 - s))quad[2].coord +
                                    ((    r)*(    s))quad[3].coord + 
                                    ((1 - r)*(    s))quad[4].coord )

(tri6::QuadraticTriangle)(r, s) = Point(((2(1 - r - s) - 1)*(1 - r - s))tri6[1].coord +
                                        ((      r         )*(2r - 1   ))tri6[2].coord +
                                        ((      s         )*(2s - 1   ))tri6[3].coord +
                                        ((     4r         )*(1 - r - s))tri6[4].coord +
                                        ((     4r         )*(        s))tri6[5].coord +
                                        ((     4s         )*(1 - r - s))tri6[6].coord )

function (quad8::QuadraticQuadrilateral)(r, s)
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/2)quad8[1].coord +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/2)quad8[2].coord +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/2)quad8[3].coord +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/2)quad8[4].coord +
                              ((1 - ξ^2)*(1 - η))quad8[5].coord +
                              ((1 - η^2)*(1 + ξ))quad8[6].coord +
                              ((1 - ξ^2)*(1 + η))quad8[7].coord +
                              ((1 - η^2)*(1 - ξ))quad8[8].coord ) / 2
end

(tet::Tetrahedron)(r, s, t) = Point((1 - r - s - t)*tet[1].coord +
                                                  r*tet[2].coord + 
                                                  s*tet[3].coord + 
                                                  t*tet[4].coord )

(hex::Hexahedron)(r, s, t) = Point(((1 - r)*(1 - s)*(1 - t))hex[1].coord + 
                                   ((    r)*(1 - s)*(1 - t))hex[2].coord +  
                                   ((    r)*(    s)*(1 - t))hex[3].coord + 
                                   ((1 - r)*(    s)*(1 - t))hex[4].coord + 
                                   ((1 - r)*(1 - s)*(    t))hex[5].coord + 
                                   ((    r)*(1 - s)*(    t))hex[6].coord + 
                                   ((    r)*(    s)*(    t))hex[7].coord + 
                                   ((1 - r)*(    s)*(    t))hex[8].coord ) 

function (tet10::QuadraticTetrahedron)(r, s, t)
    u = 1 - r - s - t
    return Point(((2u-1)u)tet10[ 1].coord +
                 ((2r-1)r)tet10[ 2].coord +
                 ((2s-1)s)tet10[ 3].coord +
                 ((2t-1)t)tet10[ 4].coord +
                    (4u*r)tet10[ 5].coord +
                    (4r*s)tet10[ 6].coord +
                    (4s*u)tet10[ 7].coord +
                    (4u*t)tet10[ 8].coord +
                    (4r*t)tet10[ 9].coord +
                    (4s*t)tet10[10].coord )
end

function (hex20::QuadraticHexahedron)(r, s, t)
    ξ = 2r - 1; η = 2s - 1; ζ = 2t - 1
    return Point(((1 - ξ)*(1 - η)*(1 - ζ)*(-2 - ξ - η - ζ)/8)hex20[ 1].coord +
                 ((1 + ξ)*(1 - η)*(1 - ζ)*(-2 + ξ - η - ζ)/8)hex20[ 2].coord +
                 ((1 + ξ)*(1 + η)*(1 - ζ)*(-2 + ξ + η - ζ)/8)hex20[ 3].coord +
                 ((1 - ξ)*(1 + η)*(1 - ζ)*(-2 - ξ + η - ζ)/8)hex20[ 4].coord +
                 ((1 - ξ)*(1 - η)*(1 + ζ)*(-2 - ξ - η + ζ)/8)hex20[ 5].coord +
                 ((1 + ξ)*(1 - η)*(1 + ζ)*(-2 + ξ - η + ζ)/8)hex20[ 6].coord +
                 ((1 + ξ)*(1 + η)*(1 + ζ)*(-2 + ξ + η + ζ)/8)hex20[ 7].coord +
                 ((1 - ξ)*(1 + η)*(1 + ζ)*(-2 - ξ + η + ζ)/8)hex20[ 8].coord +
                            ((1 - ξ^2)*(1 - η  )*(1 - ζ  )/4)hex20[ 9].coord +
                            ((1 + ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[10].coord +
                            ((1 - ξ^2)*(1 + η  )*(1 - ζ  )/4)hex20[11].coord +
                            ((1 - ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[12].coord +
                            ((1 - ξ^2)*(1 - η  )*(1 + ζ  )/4)hex20[13].coord +
                            ((1 + ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[14].coord +
                            ((1 - ξ^2)*(1 + η  )*(1 + ζ  )/4)hex20[15].coord +
                            ((1 - ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[16].coord +
                            ((1 - ξ  )*(1 - η  )*(1 - ζ^2)/4)hex20[17].coord +
                            ((1 + ξ  )*(1 - η  )*(1 - ζ^2)/4)hex20[18].coord +
                            ((1 + ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[19].coord +
                            ((1 - ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[20].coord )
end
