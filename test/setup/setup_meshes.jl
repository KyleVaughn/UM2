function setup_TriangleMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri"
    vertices = [Point{2,T}(   0, 0), 
                Point{2,T}(   1, 0), 
                Point{2,T}(1//2, 1), 
                Point{2,T}(3//2, 1)] 
    polytopes = [Triangle{U}(1, 2, 3), Triangle{U}(2, 4, 3)] 
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_QuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "quad"
    vertices = [Point{2,T}(0, 0),
                Point{2,T}(1, 0),
                Point{2,T}(1, 1),
                Point{2,T}(0, 1),
                Point{2,T}(2, 0),
                Point{2,T}(2, 1)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Quadrilateral{U}(2, 5, 6, 3)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

function setup_MixedPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri_quad"
    vertices = [Point{2,T}(0, 0),
                Point{2,T}(1, 0),
                Point{2,T}(1, 1),
                Point{2,T}(0, 1),
                Point{2,T}(2, 0)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Triangle{U}(2, 5, 3)]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(name, vertices, polytopes, groups)
end

#function setup_QuadraticTriangleMesh(::Type{T}, ::Type{U}) where {T, U}
#    name = "tri6"
#    vertices = [Point{2,T}(0, 0),
#              Point{2,T}(1, 0),
#              Point{2,T}(0, 1),
#              Point{2,T}(1, 1),
#              Point{2,T}(0.5, 0),
#              Point{2,T}(0.4, 0.4),
#              Point{2,T}(1, 0.5),
#              Point{2,T}(0.5, 1)]
#    polytopes = [SVector{6,U}(1, 2, 3, 4, 5, 6), SVector{6,U}(2, 4, 3, 7, 8, 5)]
#    groups = Dict{String, BitSet}()
#    groups["A"] = BitSet([1])
#    groups["B"] = BitSet([2])
#    return QuadraticTriangleMesh{T, U}(name, vertices, polytopes, groups)
#end
#
#function setup_QuadraticQuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
#    name = "quad8"
#    vertices = [Point{2,T}(0, 0),
#              Point{2,T}(1, 0),
#              Point{2,T}(1, 1),
#              Point{2,T}(0, 1),
#              Point{2,T}(2, 0),
#              Point{2,T}(2, 1),
#              Point{2,T}(0.5, 0),
#              Point{2,T}(0.7, 0.5),
#              Point{2,T}(0.5, 1),
#              Point{2,T}(0, 0.5),
#              Point{2,T}(1.5, 0),
#              Point{2,T}(2, 0.5),
#              Point{2,T}(1.5, 1),
#             ]
#    polytopes = [SVector{8,U}(1, 2, 3, 4, 7, 8, 9, 10),
#             SVector{8,U}(2, 5, 6, 3, 11, 12, 13, 8)]
#    groups = Dict{String, Set{Int64}}()
#    groups["A"] = Set([1])
#    groups["B"] = Set([2])
#    return QuadraticQuadrilateralMesh{T, U}(name, vertices, polytopes, groups)
#end
#
#function setup_MixedQuadraticPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
#    name = "tri6_quad8"
#    vertices = [Point{2,T}(0, 0),
#              Point{2,T}(1, 0),
#              Point{2,T}(1, 1),
#              Point{2,T}(0, 1),
#              Point{2,T}(2, 0),
#              Point{2,T}(0.5, 0),
#              Point{2,T}(0.7, 0.5),
#              Point{2,T}(0.5, 1),
#              Point{2,T}(0, 0.5),
#              Point{2,T}(1.5, 0),
#              Point{2,T}(1.5, 0.5),
#             ]
#    polytopes = [SVector{8,U}(1, 2, 3, 4, 6, 7, 8, 9),
#             SVector{6,U}(2, 5, 3, 10, 11, 7)]
#    groups = Dict{String, Set{Int64}}()
#    groups["A"] = Set([1])
#    groups["B"] = Set([2])
#    return MixedQuadraticPolygonMesh{T, U}(name, vertices, polytopes, groups)
#end
