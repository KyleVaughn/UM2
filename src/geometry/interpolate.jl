# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

(l::LineSegment)(r) = Point((1-r)*l[1].coords + r*l[2].coords)

(q::QuadraticSegment)(r) = Point(((2r - 1)*( r - 1))q[1].coords + 
                                 (( r    )*(2r - 1))q[2].coords + 
                                 ((4r    )*( 1 - r))q[3].coords )

(tri::Triangle)(r, s) = Point((1 - r - s)*tri[1].coords + 
                                        r*tri[2].coords + 
                                        s*tri[3].coords )

(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1].coords + 
                                    ((    r)*(1 - s))quad[2].coords +
                                    ((    r)*(    s))quad[3].coords + 
                                    ((1 - r)*(    s))quad[4].coords )

(tri6::QuadraticTriangle)(r, s) = Point(((2(1 - r - s) - 1)*(1 - r - s))tri6[1].coords +
                                        ((      r         )*(2r - 1   ))tri6[2].coords +
                                        ((      s         )*(2s - 1   ))tri6[3].coords +
                                        ((     4r         )*(1 - r - s))tri6[4].coords +
                                        ((     4r         )*(        s))tri6[5].coords +
                                        ((     4s         )*(1 - r - s))tri6[6].coords )

function (quad8::QuadraticQuadrilateral)(r, s)
    ξ = 2r - 1; η = 2s - 1
    return Point(((1 - ξ)*(1 - η)*(-ξ - η - 1)/2)quad8[1].coords +
                 ((1 + ξ)*(1 - η)*( ξ - η - 1)/2)quad8[2].coords +
                 ((1 + ξ)*(1 + η)*( ξ + η - 1)/2)quad8[3].coords +
                 ((1 - ξ)*(1 + η)*(-ξ + η - 1)/2)quad8[4].coords +
                              ((1 - ξ^2)*(1 - η))quad8[5].coords +
                              ((1 - η^2)*(1 + ξ))quad8[6].coords +
                              ((1 - ξ^2)*(1 + η))quad8[7].coords +
                              ((1 - η^2)*(1 - ξ))quad8[8].coords ) / 2
end

(tet::Tetrahedron)(r, s, t) = Point((1 - r - s - t)*tet[1].coords +
                                                  r*tet[2].coords + 
                                                  s*tet[3].coords + 
                                                  t*tet[4].coords )

(hex::Hexahedron)(r, s, t) = Point(((1 - r)*(1 - s)*(1 - t))hex[1].coords + 
                                   ((    r)*(1 - s)*(1 - t))hex[2].coords +  
                                   ((    r)*(    s)*(1 - t))hex[3].coords + 
                                   ((1 - r)*(    s)*(1 - t))hex[4].coords + 
                                   ((1 - r)*(1 - s)*(    t))hex[5].coords + 
                                   ((    r)*(1 - s)*(    t))hex[6].coords + 
                                   ((    r)*(    s)*(    t))hex[7].coords + 
                                   ((1 - r)*(    s)*(    t))hex[8].coords ) 

function (tet10::QuadraticTetrahedron)(r, s, t)
    u = 1 - r - s - t
    return Point(((2u-1)u)tet10[ 1].coords +
                 ((2r-1)r)tet10[ 2].coords +
                 ((2s-1)s)tet10[ 3].coords +
                 ((2t-1)t)tet10[ 4].coords +
                    (4u*r)tet10[ 5].coords +
                    (4r*s)tet10[ 6].coords +
                    (4s*u)tet10[ 7].coords +
                    (4u*t)tet10[ 8].coords +
                    (4r*t)tet10[ 9].coords +
                    (4s*t)tet10[10].coords )
end

function (hex20::QuadraticHexahedron)(r, s, t)
    ξ = 2r - 1; η = 2s - 1; ζ = 2t - 1
    return Point(((1 - ξ)*(1 - η)*(1 - ζ)*(-2 - ξ - η - ζ)/8)hex20[ 1].coords +
                 ((1 + ξ)*(1 - η)*(1 - ζ)*(-2 + ξ - η - ζ)/8)hex20[ 2].coords +
                 ((1 + ξ)*(1 + η)*(1 - ζ)*(-2 + ξ + η - ζ)/8)hex20[ 3].coords +
                 ((1 - ξ)*(1 + η)*(1 - ζ)*(-2 - ξ + η - ζ)/8)hex20[ 4].coords +
                 ((1 - ξ)*(1 - η)*(1 + ζ)*(-2 - ξ - η + ζ)/8)hex20[ 5].coords +
                 ((1 + ξ)*(1 - η)*(1 + ζ)*(-2 + ξ - η + ζ)/8)hex20[ 6].coords +
                 ((1 + ξ)*(1 + η)*(1 + ζ)*(-2 + ξ + η + ζ)/8)hex20[ 7].coords +
                 ((1 - ξ)*(1 + η)*(1 + ζ)*(-2 - ξ + η + ζ)/8)hex20[ 8].coords +
                            ((1 - ξ^2)*(1 - η  )*(1 - ζ  )/4)hex20[ 9].coords +
                            ((1 + ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[10].coords +
                            ((1 - ξ^2)*(1 + η  )*(1 - ζ  )/4)hex20[11].coords +
                            ((1 - ξ  )*(1 - η^2)*(1 - ζ  )/4)hex20[12].coords +
                            ((1 - ξ^2)*(1 - η  )*(1 + ζ  )/4)hex20[13].coords +
                            ((1 + ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[14].coords +
                            ((1 - ξ^2)*(1 + η  )*(1 + ζ  )/4)hex20[15].coords +
                            ((1 - ξ  )*(1 - η^2)*(1 + ζ  )/4)hex20[16].coords +
                            ((1 - ξ  )*(1 - η  )*(1 - ζ^2)/4)hex20[17].coords +
                            ((1 + ξ  )*(1 - η  )*(1 - ζ^2)/4)hex20[18].coords +
                            ((1 + ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[19].coords +
                            ((1 - ξ  )*(1 + η  )*(1 - ζ^2)/4)hex20[20].coords )
end
