using MOCNeutronTransport
mesh_files = [("Abaqus", ".inp")]
for (file_type, ext) in mesh_files
    @testset "Import $file_type" begin
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
    
            @testset "QuadraticTriangleMesh" begin
                ref_mesh = setup_QuadraticTriangleMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/tri6"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
            @testset "QuadraticQuadrilateralMesh" begin
                ref_mesh = setup_QuadraticQuadrilateralMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/quad8"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
    
            @testset "QuadraticPolygonMesh" begin
                ref_mesh = setup_QuadraticPolygonMesh(T, UInt16)
                mesh = import_mesh("./mesh/mesh_files/tri6_quad8"*ext, T)
                @test mesh.name == ref_mesh.name
                for i in eachindex(ref_mesh.points)
                    @test mesh.points[i] ≈ ref_mesh.points[i]
                end
                @test mesh.faces == ref_mesh.faces
                @test mesh.face_sets == ref_mesh.face_sets
            end
        end
    end
end
