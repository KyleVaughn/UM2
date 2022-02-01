using MOCNeutronTransport
@testset "Abaqus2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Triangles" begin
            filepath = "./mesh/mesh_files/tri.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(0.5, 1),
                          Point2D{F}(1.5, 1)]
            ref_faces = [SVector(1, 2, 3), SVector(2, 4, 3)]
            ref_face_sets = Dict{String, Set{Int64}}()
            ref_face_sets["A"] = Set([1])
            ref_face_sets["B"] = Set([2])

            # read_abaqus2d
            mesh = read_abaqus2d(filepath, F)
            for i in eachindex(ref_points)
                @test mesh.points[i] ≈ ref_points[i]
            end
            @test typeof(mesh.points[1]) == Point2D{F}
            @test mesh.faces == ref_faces
            @test mesh.name == "tri"
            @test mesh.face_sets == ref_face_sets
        end

        @testset "Quadrilaterals" begin
            filepath = "./mesh/mesh_files/quad.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(1, 1),
                          Point2D{F}(0, 1),
                          Point2D{F}(2, 0),
                          Point2D{F}(2, 1)]
            ref_faces = [SVector(1, 2, 3, 4), SVector(2, 5, 6, 3)]
            ref_face_sets = Dict{String, Set{Int64}}()
            ref_face_sets["A"] = Set([1])
            ref_face_sets["B"] = Set([2])

            # read_abaqus2d
            mesh = read_abaqus2d(filepath, F)
            for i in eachindex(ref_points)
                @test mesh.points[i] ≈ ref_points[i]
            end
            @test typeof(mesh.points[1]) == Point2D{F}
            @test mesh.faces == ref_faces
            @test mesh.name == "quad"
            @test mesh.face_sets == ref_face_sets
        end

        @testset "QuadraticTriangles" begin
            filepath = "./mesh/mesh_files/tri6.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(0, 1),
                          Point2D{F}(1, 1),
                          Point2D{F}(0.5, 0),
                          Point2D{F}(0.4, 0.4),
                          Point2D{F}(1, 0.5),
                          Point2D{F}(0.5, 1)]
            ref_faces = [SVector(1, 2, 3, 4, 5, 6), SVector(2, 4, 3, 7, 8, 5)]
            ref_face_sets = Dict{String, Set{Int64}}()
            ref_face_sets["A"] = Set([1])
            ref_face_sets["B"] = Set([2])

            # read_abaqus2d
            mesh = read_abaqus2d(filepath, F)
            for i in eachindex(ref_points)
                @test mesh.points[i] ≈ ref_points[i]
            end
            @test typeof(mesh.points[1]) == Point2D{F}
            @test mesh.faces == ref_faces
            @test mesh.name == "tri6"
            @test mesh.face_sets == ref_face_sets
        end

        @testset "QuadraticQuadrilaterals" begin
            filepath = "./mesh/mesh_files/quad8.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(1, 1),
                          Point2D{F}(0, 1),
                          Point2D{F}(2, 0),
                          Point2D{F}(2, 1),
                          Point2D{F}(0.5, 0),
                          Point2D{F}(0.7, 0.5),
                          Point2D{F}(0.5, 1), 
                          Point2D{F}(0, 0.5), 
                          Point2D{F}(1.5, 0), 
                          Point2D{F}(2, 0.5), 
                          Point2D{F}(1.5, 1), 
                         ]
            ref_faces = [SVector(1, 2, 3, 4, 7, 8, 9, 10), 
                         SVector(2, 5, 6, 3, 11, 12, 13, 8)]
            ref_face_sets = Dict{String, Set{Int64}}()
            ref_face_sets["A"] = Set([1])
            ref_face_sets["B"] = Set([2])

            # read_abaqus2d
            mesh = read_abaqus2d(filepath, F)
            for i in eachindex(ref_points)
                @test mesh.points[i] ≈ ref_points[i]
            end
            @test typeof(mesh.points[1]) == Point2D{F}
            @test mesh.faces == ref_faces
            @test mesh.name == "quad8"
            @test mesh.face_sets == ref_face_sets
        end
    end
end
