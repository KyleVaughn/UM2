@testset "PolyTopeVertexMesh" begin
    # Simply check that constructors work as intended
    @testset "TriangleMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_TriangleMesh(T, U)
        end
    end end

    @testset "QuadrialteralMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_QuadrilateralMesh(T, U)
        end
    end end

    @testset "MixedPolygonMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_MixedPolygonMesh(T, U)
        end
    end end

    @testset "QuadraticTriangleMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_QuadraticTriangleMesh(T, U)
        end
    end end

    @testset "QuadraticQuadrialteralMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_QuadraticQuadrilateralMesh(T, U)
        end
    end end

    @testset "MixedQuadraticPolygonMesh" begin for T in Floats
        for U in [UInt16, UInt32, UInt64]
            mesh = setup_MixedQuadraticPolygonMesh(T, U)
        end
    end end
end
