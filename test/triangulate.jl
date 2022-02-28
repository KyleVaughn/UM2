@testset "Triangulate" begin
    @testset "ConvexPolygon{2}" begin
        for T in [Float32, Float64, BigFloat]
            quad = setup_Quadrilateral2D(T)
            triangles = triangulate(quad)
            @test length(triangles) == 2
            @test triangles[1].points[1] ≈ quad[1] 
            @test triangles[1].points[2] ≈ quad[2] 
            @test triangles[1].points[3] ≈ quad[3] 
            @test triangles[2].points[1] ≈ quad[1] 
            @test triangles[2].points[2] ≈ quad[3] 
            @test triangles[2].points[3] ≈ quad[4] 
        end
    end

    @testset "Quadrilateral3D" begin
        for T in [Float32, Float64, BigFloat]
            quad = setup_Quadrilateral3D(T)
            triangles = triangulate(quad, Val(0))
            @test length(triangles) == 2
            @test triangles[1].points[1] ≈ quad[1] 
            @test triangles[1].points[2] ≈ quad[2] 
            @test triangles[1].points[3] ≈ quad[3] 
            @test triangles[2].points[1] ≈ quad[3] 
            @test triangles[2].points[2] ≈ quad[4] 
            @test triangles[2].points[3] ≈ quad[1] 
        end
    end

    @testset "QuadraticTriangle" begin
        for T in [Float32, Float64, BigFloat]
            tri6 = setup_QuadraticTriangle3D(T)
            triangles = triangulate(tri6, Val(0))
            @test length(triangles) == 1
            @test triangles[1].points[1] ≈ tri6[1] 
            @test triangles[1].points[2] ≈ tri6[2] 
            @test triangles[1].points[3] ≈ tri6[3] 

            triangles = triangulate(tri6, Val(1))
            @test length(triangles) == 4
            @test triangles[1].points[1] ≈ tri6[6] 
            @test triangles[1].points[2] ≈ tri6[5] 
            @test triangles[1].points[3] ≈ tri6[3] 
            @test triangles[2].points[1] ≈ tri6[6] 
            @test triangles[2].points[2] ≈ tri6[4] 
            @test triangles[2].points[3] ≈ tri6[5] 
            @test triangles[3].points[1] ≈ tri6[1] 
            @test triangles[3].points[2] ≈ tri6[4] 
            @test triangles[3].points[3] ≈ tri6[6] 
            @test triangles[4].points[1] ≈ tri6[4] 
            @test triangles[4].points[2] ≈ tri6[2] 
            @test triangles[4].points[3] ≈ tri6[5] 
        end
    end

    @testset "QuadraticQuadrilateral" begin
        for T in [Float32, Float64, BigFloat]
            quad8 = setup_QuadraticQuadrilateral3D(T)
            triangles = triangulate(quad8, Val(0))
            @test length(triangles) == 2
            @test triangles[1].points[1] ≈ quad8[1] 
            @test triangles[1].points[2] ≈ quad8[2] 
            @test triangles[1].points[3] ≈ quad8[3] 
            @test triangles[2].points[1] ≈ quad8[3] 
            @test triangles[2].points[2] ≈ quad8[4] 
            @test triangles[2].points[3] ≈ quad8[1] 

            triangles = triangulate(quad8, Val(1))
            @test length(triangles) == 8
        end
    end
end
