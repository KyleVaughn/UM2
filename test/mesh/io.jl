mesh_files = [("Abaqus", ".inp")]
for (file_type, ext) in mesh_files
    @testset "Import $file_type" begin for T in Floats
        @testset "$T" begin
            @testset "TriangleMesh" begin
                ref_mesh = setup_Triangle_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/tri" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end

            @testset "QuadrilateralMesh" begin
                ref_mesh = setup_Quadrilateral_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/quad" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end

            @testset "MixedPolygonMesh" begin
                ref_mesh = setup_MixedPolygon_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/tri_quad" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end

            @testset "QuadraticTriangleMesh" begin
                ref_mesh = setup_QuadraticTriangle_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/tri6" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end

            @testset "QuadraticQuadrilateralMesh" begin
                ref_mesh = setup_QuadraticQuadrilateral_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/quad8" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end

            @testset "MixedQuadraticPolygonMesh" begin
                ref_mesh = setup_MixedQuadraticPolygon_VM(T, UInt8)
                mesh = import_mesh("./mesh/mesh_files/tri6_quad8" * ext, T)
                @test all(p -> getproperty(mesh, p) == getproperty(ref_mesh, p), 
                          propertynames(mesh))
            end
        end
    end end
end
