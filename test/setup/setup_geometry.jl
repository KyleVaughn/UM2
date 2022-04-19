# Reference geometry that is sufficiently complex to test relevant methods,
function setup_LineSegment2(::Type{T}) where {T}
    return LineSegment(Point{2,T}(0,0), Point{2,T}(1,1))
end

function setup_LineSegment3(::Type{T}) where {T}
    return LineSegment(Point{3,T}(0,0,0), 
                       Point{3,T}(1,1,1))
end

function setup_QuadraticSegment2(::Type{T}) where {T}
    return QuadraticSegment(Point{2,T}(0,0), 
                            Point{2,T}(1,0), 
                            Point{2,T}(1//2, 1//2))
end

function setup_QuadraticSegment3(::Type{T}) where {T}
    return QuadraticSegment(Point{3,T}(0,0,0), 
                            Point{3,T}(1,1,0), 
                            Point{3,T}(1//2, 1//2, 1//2))
end

function setup_AABox2(::Type{T}) where {T}
    return AABox(Point{2,T}(0,0), 
                 Point{2,T}(1,1))
end

function setup_AABox3(::Type{T}) where {T}
    return AABox(Point{3,T}(0,0,0), 
                 Point{3,T}(1,1,1))
end

function setup_Triangle2(::Type{T}) where {T}
    return Triangle(Point{2,T}(0,0), 
                    Point{2,T}(1,0), 
                    Point{2,T}(0,1))
end

function setup_Triangle3(::Type{T}) where {T}
    return Triangle(Point{3,T}(0,0,0), 
                    Point{3,T}(1,0,0), 
                    Point{3,T}(0,1,1))
end

function setup_Quadrilateral2(::Type{T}) where {T}
    return Quadrilateral(Point{2,T}(0,0), 
                         Point{2,T}(1,0), 
                         Point{2,T}(1,1), 
                         Point{2,T}(0,1))
end

function setup_Quadrilateral3(::Type{T}) where {T}
    return Quadrilateral(Point{3,T}(0,0,0), 
                         Point{3,T}(1,0,1), 
                         Point{3,T}(1,1,0), 
                         Point{3,T}(0,1,1))
end

function setup_QuadraticTriangle2(::Type{T}) where {T}
    return QuadraticTriangle(Point{2,T}(0,0), 
                             Point{2,T}(1,0), 
                             Point{2,T}(0,1),
                             Point{2,T}(1//2, 1//20), 
                             Point{2,T}(1//2, 1//2), 
                             Point{2,T}(1//20, 1//2))
end

function setup_QuadraticTriangle3(::Type{T}) where {T}
    return QuadraticTriangle(Point{3,T}(0,0,0), 
                             Point{3,T}(1,0,0), 
                             Point{3,T}(0,1,0),
                             Point{3,T}(1//2,0,1//2), 
                             Point{3,T}(1//2,1//2,0), 
                             Point{3,T}(0,1//2,1//2))
end

function setup_QuadraticQuadrilateral2(::Type{T}) where {T}
    return QuadraticQuadrilateral(Point{2,T}(0,0), 
                                  Point{2,T}(1,0), 
                                  Point{2,T}(1,1), 
                                  Point{2,T}(0,1),
                                  Point{2,T}(1//2, 0), 
                                  Point{2,T}(9//10, 1//2), 
                                  Point{2,T}(1//2, 1), 
                                  Point{2,T}(1//10, 1//2))
end

function setup_QuadraticQuadrilateral3(::Type{T}) where {T}
    return QuadraticQuadrilateral(Point{3,T}(0,0,0), 
                                  Point{3,T}(1,0,0), 
                                  Point{3,T}(1,1,0), 
                                  Point{3,T}(0,1,0),
                                  Point{3,T}(1//2, 0, 1//2), 
                                  Point{3,T}(9//10, 1//2, 0), 
                                  Point{3,T}(1//2, 1, 1//2), 
                                  Point{3,T}(1//10, 1//2, 0))
end

function setup_Tetrahedron(::Type{T}) where {T}
    return Tetrahedron(Point{3,T}(0, 0, 0), 
                       Point{3,T}(1, 0, 0),
                       Point{3,T}(0, 1, 0), 
                       Point{3,T}(0, 0, 1))
end

function setup_Hexahedron(::Type{T}) where {T}
    return Hexahedron(Point{3,T}(0, 0, 0),
                      Point{3,T}(1, 0, 0),
                      Point{3,T}(0, 1, 0),
                      Point{3,T}(1, 1, 0),
                      Point{3,T}(0, 0, 1),
                      Point{3,T}(1, 0, 1),
                      Point{3,T}(0, 1, 1),
                      Point{3,T}(1, 1, 1))
end

function setup_QuadraticTetrahedron(::Type{T}) where {T}
    return QuadraticTetrahedron(Point{3,T}(0, 0, 0),
                                Point{3,T}(1, 0, 0),
                                Point{3,T}(0, 1, 0),
                                Point{3,T}(0, 0, 1),
                                Point{3,T}(1//2,    0,    0),
                                Point{3,T}(1//2, 1//2,    0),  
                                Point{3,T}(0,    1//2,    0),  
                                Point{3,T}(0,       0, 1//2),
                                Point{3,T}(1//2,    0, 1//2),
                                Point{3,T}(0,    1//2, 1//2))
end

function setup_QuadraticHexahedron(::Type{T}) where {T}
    return QuadraticHexahedron(Point{3,T}(0, 0, 0),
                               Point{3,T}(1, 0, 0),
                               Point{3,T}(0, 1, 0),
                               Point{3,T}(0, 0, 1),
                               Point{3,T}(1//2,    0,    0),
                               Point{3,T}(1//2, 1//2,    0),
                               Point{3,T}(0,    1//2,    0),
                               Point{3,T}(0,       0, 1//2),
                               Point{3,T}(1//2,    0, 1//2),
                               Point{3,T}(0,    1//2, 1//2),
                               Point{3,T}(0, 0, 0),
                               Point{3,T}(1, 0, 0),
                               Point{3,T}(0, 1, 0),
                               Point{3,T}(0, 0, 1),
                               Point{3,T}(1//2,    0,    0),
                               Point{3,T}(1//2, 1//2,    0),
                               Point{3,T}(0,    1//2,    0),
                               Point{3,T}(0,       0, 1//2),
                               Point{3,T}(1//2,    0, 1//2),
                               Point{3,T}(0,    1//2, 1//2))
end
