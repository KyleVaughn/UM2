# Setup geometry in the unit box for tests
# The geometry should be sufficiently complex to test relevant methods,

function unit_LineSegment2D(::Type{T}) where {T}
    return LineSegment(Point2D{T}(0,0), Point2D{T}(1,1))
end

function unit_LineSegment3D(::Type{T}) where {T}
    return LineSegment(Point3D{T}(0,0,0), Point3D{T}(1,1,1))
end

function unit_QuadraticSegment2D(::Type{T}) where {T}
    return QuadraticSegment(Point2D{T}(0,0), Point2D{T}(1,0), Point2D{T}(1//2, 1//2))
end

function unit_QuadraticSegment3D(::Type{T}) where {T}
    return QuadraticSegment(Point3D{T}(0,0,0), 
                            Point3D{T}(1,1,0), 
                            Point3D{T}(1//2, 1//2, 1//2))
end

function unit_AABox2D(::Type{T}) where {T}
    return AABox(Point2D{T}(0,0), Point2D{T}(1,1))
end

function unit_AABox3D(::Type{T}) where {T}
    return AABox(Point3D{T}(0,0,0), Point3D{T}(1,1,1))
end

function unit_Triangle2D(::Type{T}) where {T}
    return Triangle(Point2D{T}(0,0), Point2D{T}(1,0), Point2D{T}(0,1))
end

function unit_Triangle3D(::Type{T}) where {T}
    return Triangle(Point3D{T}(0,0,0), Point3D{T}(1,0,0), Point3D{T}(0,1,1))
end

function unit_Quadrilateral2D(::Type{T}) where {T}
    return Quadrilateral(Point2D{T}(0,0), Point2D{T}(1,0), 
                         Point2D{T}(1,1), Point2D{T}(0,1))
end

function unit_Quadrilateral3D(::Type{T}) where {T}
    return Quadrilateral(Point3D{T}(0,0,0), Point3D{T}(1,0,1), 
                         Point3D{T}(1,1,0), Point3D{T}(0,1,1))
end

function unit_QuadraticTriangle2D(::Type{T}) where {T}
    return QuadraticTriangle(Point2D{T}(0,0), Point2D{T}(1,0), Point2D{T}(0,1),
                             Point2D{T}(1//2, 1//20), Point2D{T}(1//2, 1//2), 
                             Point2D{T}(1//20, 1//2))
end

function unit_QuadraticTriangle3D(::Type{T}) where {T}
    return QuadraticTriangle(Point3D{T}(0,0,0), Point3D{T}(1,0,0), Point3D{T}(0,1,0),
                             Point3D{T}(1//2,0,1//2), Point3D{T}(1//2,0,0), 
                             Point3D{T}(0,1,1//2))
end
