@testset "PolygonMesh" begin
    @testset "TriangleMesh" begin
        for T in [Float32, Float64, BigFloat]
            for U in [UInt16, UInt32, UInt64]
                name = "tri"
                points = [Point2D{T}(0, 0),
                          Point2D{T}(1, 0),
                          Point2D{T}(0.5, 1),
                          Point2D{T}(1.5, 1)]
                faces = [SVector{3, U}(1, 2, 3), SVector{3, U}(2, 4, 3)]
                face_sets = Dict{String, BitSet}()
                face_sets["A"] = BitSet([1])
                face_sets["B"] = BitSet([2])
                mesh = TriangleMesh(name, points, faces, face_sets)
                @test mesh.name == name
                @test mesh.points == points
                @test mesh.faces == faces
                @test mesh.face_sets == face_sets
            end
        end
    end

#
#
#
#
#function setup_TriangleMesh(::Type{T}, ::Type{U}) where {T, U}









#    return TriangleMesh{T, U}(name, points, faces, face_sets)
#end
#
#function setup_QuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
#    name = "quad"
#    points = [Point2D{T}(0, 0),
#              Point2D{T}(1, 0),
#              Point2D{T}(1, 1),
#              Point2D{T}(0, 1),
#              Point2D{T}(2, 0),
#              Point2D{T}(2, 1)]
#    faces = [SVector{4,U}(1, 2, 3, 4), SVector{4,U}(2, 5, 6, 3)]
#    face_sets = Dict{String, BitSet}()
#    face_sets["A"] = BitSet([1])
#    face_sets["B"] = BitSet([2])
#    return QuadrilateralMesh{T, U}(name, points, faces, face_sets)
#end
#
#function setup_ConvexPolygonMesh(::Type{T}, ::Type{U}) where {T, U}
#    name = "tri_quad"
#    points = [Point2D{T}(0, 0),
#              Point2D{T}(1, 0),
#              Point2D{T}(1, 1),
#              Point2D{T}(0, 1),
#              Point2D{T}(2, 0)]
#    faces = [SVector{4,U}(1, 2, 3, 4), SVector{3,U}(2, 5, 3)]
#    face_sets = Dict{String, BitSet}()
#    face_sets["A"] = BitSet([1])
#    face_sets["B"] = BitSet([2])
#    return ConvexPolygonMesh{T, U}(name, points, faces, face_sets)
#end
end
