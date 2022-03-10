function setup_TriangleMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri"
    points = [Point2D{T}(0, 0), 
              Point2D{T}(1, 0), 
              Point2D{T}(0.5, 1), 
              Point2D{T}(1.5, 1)] 
    faces = [SVector{3, U}(1, 2, 3), SVector{3, U}(2, 4, 3)] 
    face_sets = Dict{String, BitSet}()
    face_sets["A"] = BitSet([1])
    face_sets["B"] = BitSet([2])
    return TriangleMesh{T, U}(name, points, faces, face_sets)
end

function setup_QuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "quad"
    points = [Point2D{T}(0, 0),
              Point2D{T}(1, 0),
              Point2D{T}(1, 1),
              Point2D{T}(0, 1),
              Point2D{T}(2, 0),
              Point2D{T}(2, 1)]
    faces = [SVector{4,U}(1, 2, 3, 4), SVector{4,U}(2, 5, 6, 3)]
    face_sets = Dict{String, BitSet}()
    face_sets["A"] = BitSet([1])
    face_sets["B"] = BitSet([2])
    return QuadrilateralMesh{T, U}(name, points, faces, face_sets)
end

function setup_ConvexPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri_quad"
    points = [Point2D{T}(0, 0),
              Point2D{T}(1, 0),
              Point2D{T}(1, 1),
              Point2D{T}(0, 1),
              Point2D{T}(2, 0)]
    faces = [SVector{4,U}(1, 2, 3, 4), SVector{3,U}(2, 5, 3)]
    face_sets = Dict{String, BitSet}()
    face_sets["A"] = BitSet([1])
    face_sets["B"] = BitSet([2])
    return ConvexPolygonMesh{T, U}(name, points, faces, face_sets)
end

function setup_QuadraticTriangleMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6"
    points = [Point2D{T}(0, 0),
              Point2D{T}(1, 0),
              Point2D{T}(0, 1),
              Point2D{T}(1, 1),
              Point2D{T}(0.5, 0),
              Point2D{T}(0.4, 0.4),
              Point2D{T}(1, 0.5),
              Point2D{T}(0.5, 1)]
    faces = [SVector{6,U}(1, 2, 3, 4, 5, 6), SVector{6,U}(2, 4, 3, 7, 8, 5)]
    face_sets = Dict{String, BitSet}()
    face_sets["A"] = BitSet([1])
    face_sets["B"] = BitSet([2])
    return QuadraticTriangleMesh{T, U}(name, points, faces, face_sets)
end

function setup_QuadraticQuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "quad8"
    points = [Point2D{T}(0, 0),
              Point2D{T}(1, 0),
              Point2D{T}(1, 1),
              Point2D{T}(0, 1),
              Point2D{T}(2, 0),
              Point2D{T}(2, 1),
              Point2D{T}(0.5, 0),
              Point2D{T}(0.7, 0.5),
              Point2D{T}(0.5, 1),
              Point2D{T}(0, 0.5),
              Point2D{T}(1.5, 0),
              Point2D{T}(2, 0.5),
              Point2D{T}(1.5, 1),
             ]
    faces = [SVector{8,U}(1, 2, 3, 4, 7, 8, 9, 10),
             SVector{8,U}(2, 5, 6, 3, 11, 12, 13, 8)]
    face_sets = Dict{String, Set{Int64}}()
    face_sets["A"] = Set([1])
    face_sets["B"] = Set([2])
    return QuadraticQuadrilateralMesh{T, U}(name, points, faces, face_sets)
end

function setup_QuadraticPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6_quad8"
    points = [Point2D{T}(0, 0),
              Point2D{T}(1, 0),
              Point2D{T}(1, 1),
              Point2D{T}(0, 1),
              Point2D{T}(2, 0),
              Point2D{T}(0.5, 0),
              Point2D{T}(0.7, 0.5),
              Point2D{T}(0.5, 1),
              Point2D{T}(0, 0.5),
              Point2D{T}(1.5, 0),
              Point2D{T}(1.5, 0.5),
             ]
    faces = [SVector{8,U}(1, 2, 3, 4, 6, 7, 8, 9),
             SVector{6,U}(2, 5, 3, 10, 11, 7)]
    face_sets = Dict{String, Set{Int64}}()
    face_sets["A"] = Set([1])
    face_sets["B"] = Set([2])
    return QuadraticPolygonMesh{T, U}(name, points, faces, face_sets)
end
