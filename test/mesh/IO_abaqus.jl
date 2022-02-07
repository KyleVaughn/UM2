using MOCNeutronTransport
@testset "Abaqus2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Triangles" begin
            filepath = "./mesh/mesh_files/tri.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(0.5, 1),
                          Point2D{F}(1.5, 1)]
            ref_edges = [SVector(1, 2),
                         SVector(1, 3),
                         SVector(2, 3),
                         SVector(2, 4),
                         SVector(3, 4)]
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
            @test mesh.edges == ref_edges
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
            ref_edges = [SVector(1, 2),
                         SVector(1, 4),
                         SVector(2, 3),
                         SVector(2, 5),
                         SVector(3, 4),
                         SVector(3, 6),
                         SVector(5, 6)]
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
            @test mesh.edges == ref_edges
            @test mesh.name == "quad"
            @test mesh.face_sets == ref_face_sets
        end

        @testset "Polygons" begin
            filepath = "./mesh/mesh_files/tri_quad.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(1, 1),
                          Point2D{F}(0, 1),
                          Point2D{F}(2, 0)]
            ref_edges = [SVector(1, 2),
                         SVector(1, 4),
                         SVector(2, 3),
                         SVector(2, 5),
                         SVector(3, 4),
                         SVector(3, 5)]
            ref_faces = [SVector(1, 2, 3, 4), SVector(2, 5, 3)]
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
            @test mesh.edges == ref_edges
            @test mesh.name == "tri_quad"
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
            ref_edges = [SVector(1, 2, 4),
                         SVector(1, 3, 6),
                         SVector(2, 3, 5),
                         SVector(2, 4, 7),
                         SVector(3, 4, 8)]
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
            @test mesh.edges == ref_edges
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
            ref_edges = [SVector(1, 2, 7 ),
                         SVector(1, 4, 10),
                         SVector(2, 3, 8 ),
                         SVector(2, 5, 11),
                         SVector(3, 4, 9 ),
                         SVector(3, 6, 13),
                         SVector(5, 6, 12)]
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
            @test mesh.edges == ref_edges
            @test mesh.name == "quad8"
            @test mesh.face_sets == ref_face_sets
        end

        @testset "QuadraticPolygons" begin
            filepath = "./mesh/mesh_files/tri6_quad8.inp"
            ref_points = [Point2D{F}(0, 0),
                          Point2D{F}(1, 0),
                          Point2D{F}(1, 1),
                          Point2D{F}(0, 1),
                          Point2D{F}(2, 0),
                          Point2D{F}(0.5, 0),
                          Point2D{F}(0.7, 0.5),
                          Point2D{F}(0.5, 1), 
                          Point2D{F}(0, 0.5), 
                          Point2D{F}(1.5, 0), 
                          Point2D{F}(1.5, 0.5), 
                         ]
            ref_edges = [SVector(1, 2, 6 ),
                         SVector(1, 4, 9),
                         SVector(2, 3, 7 ),
                         SVector(2, 5, 10),
                         SVector(3, 4, 8 ),
                         SVector(3, 5, 11)]
            ref_faces = [SVector(1, 2, 3, 4, 6, 7, 8, 9), 
                         SVector(2, 5, 3, 10, 11, 7)]
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
            @test mesh.edges == ref_edges
            @test mesh.name == "tri6_quad8"
            @test mesh.face_sets == ref_face_sets
        end
    end
end
