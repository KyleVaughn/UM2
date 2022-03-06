# Reference geometry that is sufficiently complex to test relevant methods,
function setup_LineSegment2D(::Type{T}) where {T}
    return LineSegment(Point2D{T}(0,0), 
                       Point2D{T}(1,1))
end

function setup_LineSegment3D(::Type{T}) where {T}
    return LineSegment(Point3D{T}(0,0,0), 
                       Point3D{T}(1,1,1))
end

function setup_QuadraticSegment2D(::Type{T}) where {T}
    return QuadraticSegment(Point2D{T}(0,0), 
                            Point2D{T}(1,0), 
                            Point2D{T}(1//2, 1//2))
end

function setup_QuadraticSegment3D(::Type{T}) where {T}
    return QuadraticSegment(Point3D{T}(0,0,0), 
                            Point3D{T}(1,1,0), 
                            Point3D{T}(1//2, 1//2, 1//2))
end

function setup_AABox2D(::Type{T}) where {T}
    return AABox(Point2D{T}(0,0), 
                 Point2D{T}(1,1))
end

function setup_AABox3D(::Type{T}) where {T}
    return AABox(Point3D{T}(0,0,0), 
                 Point3D{T}(1,1,1))
end

function setup_Triangle2D(::Type{T}) where {T}
    return Triangle(Point2D{T}(0,0), 
                    Point2D{T}(1,0), 
                    Point2D{T}(0,1))
end

function setup_Triangle3D(::Type{T}) where {T}
    return Triangle(Point3D{T}(0,0,0), 
                    Point3D{T}(1,0,0), 
                    Point3D{T}(0,1,1))
end

function setup_Quadrilateral2D(::Type{T}) where {T}
    return Quadrilateral(Point2D{T}(0,0), 
                         Point2D{T}(1,0), 
                         Point2D{T}(1,1), 
                         Point2D{T}(0,1))
end

function setup_Quadrilateral3D(::Type{T}) where {T}
    return Quadrilateral(Point3D{T}(0,0,0), 
                         Point3D{T}(1,0,1), 
                         Point3D{T}(1,1,0), 
                         Point3D{T}(0,1,1))
end

function setup_QuadraticTriangle2D(::Type{T}) where {T}
    return QuadraticTriangle(Point2D{T}(0,0), 
                             Point2D{T}(1,0), 
                             Point2D{T}(0,1),
                             Point2D{T}(1//2, 1//20), 
                             Point2D{T}(1//2, 1//2), 
                             Point2D{T}(1//20, 1//2))
end

function setup_QuadraticTriangle3D(::Type{T}) where {T}
    return QuadraticTriangle(Point3D{T}(0,0,0), 
                             Point3D{T}(1,0,0), 
                             Point3D{T}(0,1,0),
                             Point3D{T}(1//2,0,1//2), 
                             Point3D{T}(1//2,1//2,0), 
                             Point3D{T}(0,1//2,1//2))
end

function setup_QuadraticQuadrilateral2D(::Type{T}) where {T}
    return QuadraticQuadrilateral(Point2D{T}(0,0), 
                                  Point2D{T}(1,0), 
                                  Point2D{T}(1,1), 
                                  Point2D{T}(0,1),
                                  Point2D{T}(1//2, 0), 
                                  Point2D{T}(9//10, 1//2), 
                                  Point2D{T}(1//2, 1), 
                                  Point2D{T}(1//10, 1//2))
end

function setup_QuadraticQuadrilateral3D(::Type{T}) where {T}
    return QuadraticQuadrilateral(Point3D{T}(0,0,0), 
                                  Point3D{T}(1,0,0), 
                                  Point3D{T}(1,1,0), 
                                  Point3D{T}(0,1,0),
                                  Point3D{T}(1//2, 0, 1//2), 
                                  Point3D{T}(9//10, 1//2, 0), 
                                  Point3D{T}(1//2, 1, 1//2), 
                                  Point3D{T}(1//10, 1//2, 0))
end

function setup_Tetrahedron(::Type{T}) where {T}
    return Tetrahedron(Point3D{T}(0, 0, 0), 
                       Point3D{T}(1, 0, 0),
                       Point3D{T}(0, 1, 0), 
                       Point3D{T}(0, 0, 1))
end

function setup_Hexahedron(::Type{T}) where {T}
    return Hexahedron(Point3D{T}(0, 0, 0),
                      Point3D{T}(1, 0, 0),
                      Point3D{T}(0, 1, 0),
                      Point3D{T}(1, 1, 0),
                      Point3D{T}(0, 0, 1),
                      Point3D{T}(1, 0, 1),
                      Point3D{T}(0, 1, 1),
                      Point3D{T}(1, 1, 1))
end

function setup_QuadraticTetrahedron(::Type{T}) where {T}
    return QuadraticTetrahedron(Point3D{T}(0, 0, 0),
                                Point3D{T}(1, 0, 0),
                                Point3D{T}(0, 1, 0),
                                Point3D{T}(0, 0, 1),
                                Point3D{T}(1//2,    0,    0),
                                Point3D{T}(1//2, 1//2,    0),  
                                Point3D{T}(0,    1//2,    0),  
                                Point3D{T}(0,       0, 1//2),
                                Point3D{T}(1//2,    0, 1//2),
                                Point3D{T}(0,    1//2, 1//2))
end

function setup_QuadraticHexahedron(::Type{T}) where {T}
    return QuadraticHexahedron(Point3D{T}(0, 0, 0),
                               Point3D{T}(1, 0, 0),
                               Point3D{T}(0, 1, 0),
                               Point3D{T}(0, 0, 1),
                               Point3D{T}(1//2,    0,    0),
                               Point3D{T}(1//2, 1//2,    0),
                               Point3D{T}(0,    1//2,    0),
                               Point3D{T}(0,       0, 1//2),
                               Point3D{T}(1//2,    0, 1//2),
                               Point3D{T}(0,    1//2, 1//2),
                               Point3D{T}(0, 0, 0),
                               Point3D{T}(1, 0, 0),
                               Point3D{T}(0, 1, 0),
                               Point3D{T}(0, 0, 1),
                               Point3D{T}(1//2,    0,    0),
                               Point3D{T}(1//2, 1//2,    0),
                               Point3D{T}(0,    1//2,    0),
                               Point3D{T}(0,       0, 1//2),
                               Point3D{T}(1//2,    0, 1//2),
                               Point3D{T}(0,    1//2, 1//2))
end
