@testset "Triangulate" begin
    @testset "Polygon - 2D" begin for T in Floats
        quad = setup_Quadrilateral2(T)
        triangles = triangulate(quad)
        @test length(triangles) == 2
        @test triangles[1].vertices[1] ≈ quad[1]
        @test triangles[1].vertices[2] ≈ quad[2]
        @test triangles[1].vertices[3] ≈ quad[3]
        @test triangles[2].vertices[1] ≈ quad[1]
        @test triangles[2].vertices[2] ≈ quad[3]
        @test triangles[2].vertices[3] ≈ quad[4]
    end end

    @testset "Quadrilateral - 3D" begin for T in Floats
        quad = setup_Quadrilateral3(T)
        triangles = triangulate(quad, Val(0))
        @test length(triangles) == 2
        @test triangles[1].vertices[1] ≈ quad[1]
        @test triangles[1].vertices[2] ≈ quad[2]
        @test triangles[1].vertices[3] ≈ quad[3]
        @test triangles[2].vertices[1] ≈ quad[3]
        @test triangles[2].vertices[2] ≈ quad[4]
        @test triangles[2].vertices[3] ≈ quad[1]
    end end

    @testset "QuadraticTriangle" begin for T in Floats
        tri6 = setup_QuadraticTriangle3(T)
        triangles = triangulate(tri6, Val(0))
        @test length(triangles) == 1
        @test triangles[1].vertices[1] ≈ tri6[1]
        @test triangles[1].vertices[2] ≈ tri6[2]
        @test triangles[1].vertices[3] ≈ tri6[3]

        triangles = triangulate(tri6, Val(1))
        @test length(triangles) == 4
        @test triangles[1].vertices[1] ≈ tri6[6]
        @test triangles[1].vertices[2] ≈ tri6[5]
        @test triangles[1].vertices[3] ≈ tri6[3]
        @test triangles[2].vertices[1] ≈ tri6[6]
        @test triangles[2].vertices[2] ≈ tri6[4]
        @test triangles[2].vertices[3] ≈ tri6[5]
        @test triangles[3].vertices[1] ≈ tri6[1]
        @test triangles[3].vertices[2] ≈ tri6[4]
        @test triangles[3].vertices[3] ≈ tri6[6]
        @test triangles[4].vertices[1] ≈ tri6[4]
        @test triangles[4].vertices[2] ≈ tri6[2]
        @test triangles[4].vertices[3] ≈ tri6[5]
    end end

    @testset "QuadraticQuadrilateral" begin for T in Floats
        quad8 = setup_QuadraticQuadrilateral3(T)
        triangles = triangulate(quad8, Val(0))
        @test length(triangles) == 2
        @test triangles[1].vertices[1] ≈ quad8[1]
        @test triangles[1].vertices[2] ≈ quad8[2]
        @test triangles[1].vertices[3] ≈ quad8[3]
        @test triangles[2].vertices[1] ≈ quad8[3]
        @test triangles[2].vertices[2] ≈ quad8[4]
        @test triangles[2].vertices[3] ≈ quad8[1]

        triangles = triangulate(quad8, Val(1))
        @test length(triangles) == 8
    end end
end
