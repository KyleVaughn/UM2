export faces

# Turn off the JuliaFormatter
#! format: off

function faces(p::Tetrahedron{T}) where T
    v = vertices(p)
    return Vec{4, Triangle{T}}(Triangle{T}(v[1], v[2], v[3]),
                               Triangle{T}(v[1], v[2], v[4]),
                               Triangle{T}(v[2], v[3], v[4]),
                               Triangle{T}(v[3], v[1], v[4]))
end

function faces(p::Hexahedron{T}) where T
    v = vertices(p)
    return Vec{6, Quadrilateral{T}}(Quadrilateral{T}(v[1], v[2], v[3], v[4]),
                                    Quadrilateral{T}(v[5], v[6], v[7], v[8]),
                                    Quadrilateral{T}(v[1], v[2], v[6], v[5]),
                                    Quadrilateral{T}(v[2], v[3], v[7], v[6]),
                                    Quadrilateral{T}(v[3], v[4], v[8], v[7]),
                                    Quadrilateral{T}(v[4], v[1], v[5], v[8]))
end

function faces(p::QuadraticTetrahedron{T}) where T
    v = vertices(p)
    return Vec{4, QuadraticTriangle{T}}(
                QuadraticTriangle{T}(v[1], v[2], v[3], v[5], v[ 6], v[ 7]),
                QuadraticTriangle{T}(v[1], v[2], v[4], v[5], v[ 9], v[ 8]),
                QuadraticTriangle{T}(v[2], v[3], v[4], v[6], v[10], v[ 9]),
                QuadraticTriangle{T}(v[3], v[1], v[4], v[7], v[ 8], v[10])
              )
end

function faces(p::QuadraticHexahedron{T}) where T
    v = vertices(p)
    return Vec{6, QuadraticQuadrilateral{T}}(
                QuadraticQuadrilateral{T}(v[1], v[2], v[3], v[4], v[ 9], v[10], v[11], v[12]),
                QuadraticQuadrilateral{T}(v[5], v[6], v[7], v[8], v[13], v[14], v[15], v[16]),
                QuadraticQuadrilateral{T}(v[1], v[2], v[6], v[5], v[ 9], v[18], v[13], v[17]),
                QuadraticQuadrilateral{T}(v[2], v[3], v[7], v[6], v[10], v[19], v[14], v[18]),
                QuadraticQuadrilateral{T}(v[3], v[4], v[8], v[7], v[11], v[20], v[15], v[19]),
                QuadraticQuadrilateral{T}(v[4], v[1], v[5], v[8], v[12], v[17], v[16], v[20])
               )
end
