export edge, edges

#function edges(p::Tetrahedron{T}) where {T}
#    v = vertices(p)
#    return Vec(LineSegment{T}(v[1], v[2]),
#               LineSegment{T}(v[2], v[3]),
#               LineSegment{T}(v[3], v[1]),
#               LineSegment{T}(v[1], v[4]),
#               LineSegment{T}(v[2], v[4]),
#               LineSegment{T}(v[3], v[4]))
#end
#
#function edges(p::Hexahedron{T}) where {T}
#    v = vertices(p)
#    return Vec(LineSegment{T}(v[1], v[2]), # lower z
#               LineSegment{T}(v[2], v[3]),
#               LineSegment{T}(v[3], v[4]),
#               LineSegment{T}(v[4], v[1]),
#               LineSegment{T}(v[5], v[6]), # upper z
#               LineSegment{T}(v[6], v[7]),
#               LineSegment{T}(v[7], v[8]),
#               LineSegment{T}(v[8], v[5]),
#               LineSegment{T}(v[1], v[5]), # lower, upper connections
#               LineSegment{T}(v[2], v[6]),
#               LineSegment{T}(v[3], v[7]),
#               LineSegment{T}(v[4], v[8]))
#end
#
#function edges(p::QuadraticTetrahedron{T}) where {T}
#    v = vertices(p)
#    return Vec(QuadraticSegment{T}(v[1], v[2], v[5]),
#               QuadraticSegment{T}(v[2], v[3], v[6]),
#               QuadraticSegment{T}(v[3], v[1], v[7]),
#               QuadraticSegment{T}(v[1], v[4], v[8]),
#               QuadraticSegment{T}(v[2], v[4], v[9]),
#               QuadraticSegment{T}(v[3], v[4], v[10]))
#end
#
#function edges(p::QuadraticHexahedron{T}) where {T}
#    v = vertices(p)
#    return Vec(QuadraticSegment{T}(v[1], v[2], v[9]), # lower z
#               QuadraticSegment{T}(v[2], v[3], v[10]),
#               QuadraticSegment{T}(v[3], v[4], v[11]),
#               QuadraticSegment{T}(v[4], v[1], v[12]),
#               QuadraticSegment{T}(v[5], v[6], v[13]), # upper z
#               QuadraticSegment{T}(v[6], v[7], v[14]),
#               QuadraticSegment{T}(v[7], v[8], v[15]),
#               QuadraticSegment{T}(v[8], v[5], v[16]),
#               QuadraticSegment{T}(v[1], v[5], v[17]), # lower, upper connections
#               QuadraticSegment{T}(v[2], v[6], v[18]),
#               QuadraticSegment{T}(v[3], v[7], v[19]),
#               QuadraticSegment{T}(v[4], v[8], v[20]))
#end
