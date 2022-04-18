# Reference geometry that is sufficiently complex to test relevant methods,
function setup_LineSegment2(::Type{P}) where {P<:Point}
    return LineSegment(P(0,0), P(1,1))
end

function setup_LineSegment3(::Type{P}) where {P<:Point}
    return LineSegment(P{T}(0,0,0), 
                       P{T}(1,1,1))
end

function setup_QuadraticSegment2(::Type{P}) where {P<:Point}
    return QuadraticSegment(P{T}(0,0), 
                            P{T}(1,0), 
                            P{T}(1//2, 1//2))
end

function setup_QuadraticSegment3(::Type{P}) where {P<:Point}
    return QuadraticSegment(P{T}(0,0,0), 
                            P{T}(1,1,0), 
                            P{T}(1//2, 1//2, 1//2))
end

function setup_AABox2(::Type{P}) where {P<:Point}
    return AABox(P{T}(0,0), 
                 P{T}(1,1))
end

function setup_AABox3(::Type{P}) where {P<:Point}
    return AABox(P{T}(0,0,0), 
                 P{T}(1,1,1))
end

function setup_Triangle2(::Type{P}) where {P<:Point}
    return Triangle(P{T}(0,0), 
                    P{T}(1,0), 
                    P{T}(0,1))
end

function setup_Triangle3(::Type{P}) where {P<:Point}
    return Triangle(P{T}(0,0,0), 
                    P{T}(1,0,0), 
                    P{T}(0,1,1))
end

function setup_Quadrilateral2(::Type{P}) where {P<:Point}
    return Quadrilateral(P{T}(0,0), 
                         P{T}(1,0), 
                         P{T}(1,1), 
                         P{T}(0,1))
end

function setup_Quadrilateral3(::Type{P}) where {P<:Point}
    return Quadrilateral(P{T}(0,0,0), 
                         P{T}(1,0,1), 
                         P{T}(1,1,0), 
                         P{T}(0,1,1))
end

function setup_QuadraticTriangle2(::Type{P}) where {P<:Point}
    return QuadraticTriangle(P{T}(0,0), 
                             P{T}(1,0), 
                             P{T}(0,1),
                             P{T}(1//2, 1//20), 
                             P{T}(1//2, 1//2), 
                             P{T}(1//20, 1//2))
end

function setup_QuadraticTriangle3(::Type{P}) where {P<:Point}
    return QuadraticTriangle(P{T}(0,0,0), 
                             P{T}(1,0,0), 
                             P{T}(0,1,0),
                             P{T}(1//2,0,1//2), 
                             P{T}(1//2,1//2,0), 
                             P{T}(0,1//2,1//2))
end

function setup_QuadraticQuadrilateral2(::Type{P}) where {P<:Point}
    return QuadraticQuadrilateral(P{T}(0,0), 
                                  P{T}(1,0), 
                                  P{T}(1,1), 
                                  P{T}(0,1),
                                  P{T}(1//2, 0), 
                                  P{T}(9//10, 1//2), 
                                  P{T}(1//2, 1), 
                                  P{T}(1//10, 1//2))
end

function setup_QuadraticQuadrilateral3(::Type{P}) where {P<:Point}
    return QuadraticQuadrilateral(P{T}(0,0,0), 
                                  P{T}(1,0,0), 
                                  P{T}(1,1,0), 
                                  P{T}(0,1,0),
                                  P{T}(1//2, 0, 1//2), 
                                  P{T}(9//10, 1//2, 0), 
                                  P{T}(1//2, 1, 1//2), 
                                  P{T}(1//10, 1//2, 0))
end

function setup_Tetrahedron(::Type{P}) where {P<:Point}
    return Tetrahedron(P{T}(0, 0, 0), 
                       P{T}(1, 0, 0),
                       P{T}(0, 1, 0), 
                       P{T}(0, 0, 1))
end

function setup_Hexahedron(::Type{P}) where {P<:Point}
    return Hexahedron(P{T}(0, 0, 0),
                      P{T}(1, 0, 0),
                      P{T}(0, 1, 0),
                      P{T}(1, 1, 0),
                      P{T}(0, 0, 1),
                      P{T}(1, 0, 1),
                      P{T}(0, 1, 1),
                      P{T}(1, 1, 1))
end

function setup_QuadraticTetrahedron(::Type{P}) where {P<:Point}
    return QuadraticTetrahedron(P{T}(0, 0, 0),
                                P{T}(1, 0, 0),
                                P{T}(0, 1, 0),
                                P{T}(0, 0, 1),
                                P{T}(1//2,    0,    0),
                                P{T}(1//2, 1//2,    0),  
                                P{T}(0,    1//2,    0),  
                                P{T}(0,       0, 1//2),
                                P{T}(1//2,    0, 1//2),
                                P{T}(0,    1//2, 1//2))
end

function setup_QuadraticHexahedron(::Type{P}) where {P<:Point}
    return QuadraticHexahedron(P{T}(0, 0, 0),
                               P{T}(1, 0, 0),
                               P{T}(0, 1, 0),
                               P{T}(0, 0, 1),
                               P{T}(1//2,    0,    0),
                               P{T}(1//2, 1//2,    0),
                               P{T}(0,    1//2,    0),
                               P{T}(0,       0, 1//2),
                               P{T}(1//2,    0, 1//2),
                               P{T}(0,    1//2, 1//2),
                               P{T}(0, 0, 0),
                               P{T}(1, 0, 0),
                               P{T}(0, 1, 0),
                               P{T}(0, 0, 1),
                               P{T}(1//2,    0,    0),
                               P{T}(1//2, 1//2,    0),
                               P{T}(0,    1//2,    0),
                               P{T}(0,       0, 1//2),
                               P{T}(1//2,    0, 1//2),
                               P{T}(0,    1//2, 1//2))
end
