export faces

function faces(p::Tetrahedron)
    v = vertices(p)
    return Vec(Triangle(v[1], v[2], v[3]),
               Triangle(v[1], v[2], v[4]),
               Triangle(v[2], v[3], v[4]),
               Triangle(v[3], v[1], v[4]))
end

function faces(p::Hexahedron)
    v = vertices(p)
    return Vec(Quadrilateral(v[1], v[2], v[3], v[4]),
               Quadrilateral(v[5], v[6], v[7], v[8]),
               Quadrilateral(v[1], v[2], v[6], v[5]),
               Quadrilateral(v[2], v[3], v[7], v[6]),
               Quadrilateral(v[3], v[4], v[8], v[7]),
               Quadrilateral(v[4], v[1], v[5], v[8]))
end

function faces(p::QuadraticTetrahedron)
    v = vertices(p)
    return Vec(QuadraticTriangle(v[1], v[2], v[3], v[5],  v[6],  v[7]),
               QuadraticTriangle(v[1], v[2], v[4], v[5],  v[9],  v[8]),
               QuadraticTriangle(v[2], v[3], v[4], v[6], v[10],  v[9]),
               QuadraticTriangle(v[3], v[1], v[4], v[7],  v[8], v[10]))
end

function faces(p::QuadraticHexahedron)
    v = vertices(p)
    return Vec(QuadraticQuadrilateral(v[1], v[2], v[3], v[4], v[ 9], v[10], v[11], v[12]),
               QuadraticQuadrilateral(v[5], v[6], v[7], v[8], v[13], v[14], v[15], v[16]),
               QuadraticQuadrilateral(v[1], v[2], v[6], v[5], v[ 9], v[18], v[13], v[17]),
               QuadraticQuadrilateral(v[2], v[3], v[7], v[6], v[10], v[19], v[14], v[18]),
               QuadraticQuadrilateral(v[3], v[4], v[8], v[7], v[11], v[20], v[15], v[19]),
               QuadraticQuadrilateral(v[4], v[1], v[5], v[8], v[12], v[17], v[16], v[20]))
end
