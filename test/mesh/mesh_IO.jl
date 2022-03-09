using MOCNeutronTransport
mesh_files = [("Abaqus", ".inp")]
for (file_type, ext) in mesh_files
    @testset "$file_type IO" begin
        for T in [Float32, Float64, BigFloat]
            @testset "TriangleMesh" begin
                ref_mesh = setup_TriangleMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/tri"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
            @testset "QuadrilateralMesh" begin
                ref_mesh = setup_QuadrilateralMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/quad"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
            @testset "ConvexPolygonMesh" begin
                ref_mesh = setup_ConvexPolygonMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/tri_quad"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
            @testset "QuadraticTriangles" begin
                ref_mesh = setup_QuadraticTriangleMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/tri6"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
    #        @testset "QuadraticQuadrilaterals" begin
    #            filepath = "./mesh/mesh_files/quad8.inp"
    #            ref_points = [Point2D{T}(0, 0),
    #                          Point2D{T}(1, 0),
    #                          Point2D{T}(1, 1),
    #                          Point2D{T}(0, 1),
    #                          Point2D{T}(2, 0),
    #                          Point2D{T}(2, 1),
    #                          Point2D{T}(0.5, 0),
    #                          Point2D{T}(0.7, 0.5),
    #                          Point2D{T}(0.5, 1), 
    #                          Point2D{T}(0, 0.5), 
    #                          Point2D{T}(1.5, 0), 
    #                          Point2D{T}(2, 0.5), 
    #                          Point2D{T}(1.5, 1), 
    #                         ]
    #            ref_faces = [SVector(1, 2, 3, 4, 7, 8, 9, 10), 
    #                         SVector(2, 5, 6, 3, 11, 12, 13, 8)]
    #            ref_face_sets = Dict{String, Set{Int64}}()
    #            ref_face_sets["A"] = Set([1])
    #            ref_face_sets["B"] = Set([2])
    #
    #            # read_abaqus2d
    #            mesh = read_abaqus2d(filepath, T)
    #            for i in eachindex(ref_points)
    #                @test mesh.points[i] ≈ ref_points[i]
    #            end
    #            @test typeof(mesh.points[1]) == Point2D{T}
    #            @test mesh.faces == ref_faces
    #            @test mesh.name == "quad8"
    #            @test mesh.face_sets == ref_face_sets
    #        end
    #
    #        @testset "QuadraticPolygons" begin
    #            filepath = "./mesh/mesh_files/tri6_quad8.inp"
    #            ref_points = [Point2D{T}(0, 0),
    #                          Point2D{T}(1, 0),
    #                          Point2D{T}(1, 1),
    #                          Point2D{T}(0, 1),
    #                          Point2D{T}(2, 0),
    #                          Point2D{T}(0.5, 0),
    #                          Point2D{T}(0.7, 0.5),
    #                          Point2D{T}(0.5, 1), 
    #                          Point2D{T}(0, 0.5), 
    #                          Point2D{T}(1.5, 0), 
    #                          Point2D{T}(1.5, 0.5), 
    #                         ]
    #            ref_faces = [SVector(1, 2, 3, 4, 6, 7, 8, 9), 
    #                         SVector(2, 5, 3, 10, 11, 7)]
    #            ref_face_sets = Dict{String, Set{Int64}}()
    #            ref_face_sets["A"] = Set([1])
    #            ref_face_sets["B"] = Set([2])
    #
    #            # read_abaqus2d
    #            mesh = read_abaqus2d(filepath, T)
    #            for i in eachindex(ref_points)
    #                @test mesh.points[i] ≈ ref_points[i]
    #            end
    #            @test typeof(mesh.points[1]) == Point2D{T}
    #            @test mesh.faces == ref_faces
    #            @test mesh.name == "tri6_quad8"
    #            @test mesh.face_sets == ref_face_sets
    #        end
        end
    end
end
