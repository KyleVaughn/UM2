export edges

function edges(p::Polygon{N,T}) where {N,T}
    return Vec{N, LineSegment{T}}(
                ntuple(i->LineSegment(p[ (i-1)%N + 1], 
                                      p[  i   %N + 1]
                                     ), Val(N))
               )
end

function edges(p::QuadraticPolygon{N,T}) where {N,T}
    M = N ÷ 2
    return Vec{M, QuadraticSegment{T}}(
                ntuple(i->QuadraticSegment(p[ (i-1)%M + 1], 
                                           p[  i   %M + 1],
                                           p[  i      + M]
                                          ), Val(M))
               )
end

function edges(p::Tetrahedron)
    v = vertices(p)
    return Vec(LineSegment(v[1], v[2]),
               LineSegment(v[2], v[3]),
               LineSegment(v[3], v[1]),
               LineSegment(v[1], v[4]),
               LineSegment(v[2], v[4]),
               LineSegment(v[3], v[4]))
end

function edges(p::Hexahedron)
    v = vertices(p)
    return Vec(LineSegment(v[1], v[2]), # lower z
               LineSegment(v[2], v[3]),
               LineSegment(v[3], v[4]),
               LineSegment(v[4], v[1]),
               LineSegment(v[5], v[6]), # upper z
               LineSegment(v[6], v[7]),
               LineSegment(v[7], v[8]),
               LineSegment(v[8], v[5]),
               LineSegment(v[1], v[5]), # lower, upper connections
               LineSegment(v[2], v[6]),
               LineSegment(v[3], v[7]),
               LineSegment(v[4], v[8])) 
end

function edges(p::QuadraticTetrahedron)
    v = vertices(p)
    return Vec(QuadraticSegment(v[1], v[2], v[5]),
               QuadraticSegment(v[2], v[3], v[6]),
               QuadraticSegment(v[3], v[1], v[7]),
               QuadraticSegment(v[1], v[4], v[8]),
               QuadraticSegment(v[2], v[4], v[9]),
               QuadraticSegment(v[3], v[4], v[10]))
end

function edges(p::QuadraticHexahedron)
    v = vertices(p)
    return Vec(QuadraticSegment(v[1], v[2], v[ 9]), # lower z
               QuadraticSegment(v[2], v[3], v[10]),
               QuadraticSegment(v[3], v[4], v[11]),
               QuadraticSegment(v[4], v[1], v[12]),
               QuadraticSegment(v[5], v[6], v[13]), # upper z
               QuadraticSegment(v[6], v[7], v[14]),
               QuadraticSegment(v[7], v[8], v[15]),
               QuadraticSegment(v[8], v[5], v[16]),
               QuadraticSegment(v[1], v[5], v[17]), # lower, upper connections
               QuadraticSegment(v[2], v[6], v[18]),
               QuadraticSegment(v[3], v[7], v[19]),
               QuadraticSegment(v[4], v[8], v[20])) 
end
