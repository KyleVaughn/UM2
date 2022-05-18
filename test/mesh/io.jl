mesh_files = [("Abaqus", ".inp")]
for (file_type, ext) in mesh_files
    @testset "Import $file_type" begin
        for T in Floats 
            @testset "$T" begin
                @testset "TriangleMesh" begin
                    ref_mesh = setup_TriangleMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/tri"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
    
                @testset "QuadrilateralMesh" begin
                    ref_mesh = setup_QuadrilateralMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/quad"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
    
                @testset "MixedPolygonMesh" begin
                    ref_mesh = setup_MixedPolygonMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/tri_quad"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
    
                @testset "QuadraticTriangleMesh" begin
                    ref_mesh = setup_QuadraticTriangleMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/tri6"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
    
                @testset "QuadraticQuadrilateralMesh" begin
                    ref_mesh = setup_QuadraticQuadrilateralMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/quad8"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
    
                @testset "MixedQuadraticPolygonMesh" begin
                    ref_mesh = setup_MixedQuadraticPolygonMesh(T, UInt8)
                    mesh = import_mesh("./mesh/mesh_files/tri6_quad8"*ext, T)
                    @test mesh.name == ref_mesh.name
                    for i in eachindex(ref_mesh.vertices)
                        @test mesh.vertices[i] ≈ ref_mesh.vertices[i]
                    end
                    @test mesh.polytopes == ref_mesh.polytopes
                    @test mesh.groups == ref_mesh.groups
                end
            end
        end
    end
end
