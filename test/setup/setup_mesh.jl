function setup_PVM_TriangleMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(3 // 2, 1)]
    polytopes = [Triangle{U}(1, 2, 3), Triangle{U}(2, 4, 3)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_PVM_QuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "quad"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Quadrilateral{U}(2, 5, 6, 3)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_PVM_MixedPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri_quad"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Triangle{U}(2, 5, 3)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_PVM_QuadraticTriangleMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(0, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(2 // 5, 2 // 5),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(1, 1),
        Point{2, T}(1, 1 // 2),
        Point{2, T}(1 // 2, 1)]
    polytopes = [QuadraticTriangle{U}(1, 2, 3, 4, 5, 6),
        QuadraticTriangle{U}(2, 7, 3, 8, 9, 5)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_PVM_QuadraticQuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "quad8"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(7 // 10, 1 // 2),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(2, 1 // 2),
        Point{2, T}(3 // 2, 1),
    ]
    polytopes = [QuadraticQuadrilateral{U}(1, 2, 3, 4, 7, 8, 9, 10),
        QuadraticQuadrilateral{U}(2, 5, 6, 3, 11, 12, 13, 8)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groupPoint{2, T}(1//2, 1//2)s["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_PVM_MixedQuadraticPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6_quad8"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(7 // 10, 1 // 2),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(3 // 2, 1 // 2),
    ]
    polytopes = [QuadraticQuadrilateral{U}(1, 2, 3, 4, 6, 7, 8, 9),
        QuadraticTriangle{U}(2, 5, 3, 10, 11, 7)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end
