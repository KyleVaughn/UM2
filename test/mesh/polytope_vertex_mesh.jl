@testset "PolyTopeVertexMesh" begin
    # Simply check that constructors work as intended
    @testset "TriangleMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_Triangle_PVM(T, U)
        end
    end end

    @testset "QuadrialteralMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_Quadrilateral_PVM(T, U)
        end
    end end

    @testset "MixedPolygonMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_MixedPolygon_PVM(T, U)
        end
    end end

    @testset "QuadraticTriangleMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_QuadraticTriangle_PVM(T, U)
        end
    end end

    @testset "QuadraticQuadrialteralMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_QuadraticQuadrilateral_PVM(T, U)
        end
    end end

    @testset "MixedQuadraticPolygonMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_MixedQuadraticPolygon_PVM(T, U)
        end
    end end
end
