@testset "Measure" begin
    @testset "LineSegment" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(setup_LineSegment2D(T)) ≈ sqrt(T(2))
            @test measure(setup_LineSegment3D(T)) ≈ sqrt(T(3))
        end
    end

    @testset "QuadraticSegment" begin
        for T in [Float32, Float64, BigFloat]
            q₂ = QuadraticSegment(Point2D{T}(0,0), 
                                  Point2D{T}(1,0), 
                                  Point2D{T}(1//2, 0))
            @test measure(q₂) ≈ 1
            q₂ = setup_QuadraticSegment2D(T)
            @test abs(measure(q₂) - 1.4789428575445974) < 1e-6
            q₃ = QuadraticSegment(Point3D{T}(0,0,0), 
                                  Point3D{T}(1,0,0), 
                                  Point3D{T}(1//2,0,0))
            @test measure(q₃) ≈ 1
            q₃ = setup_QuadraticSegment3D(T)
            @test abs(measure(q₃) - 1.7978527887818835) < 1e-6
        end
    end

    @testset "AABox" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(setup_AABox2D(T)) ≈ 1
            @test measure(setup_AABox3D(T)) ≈ 1
        end
    end

    @testset "ConvexPolygon" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(setup_Triangle2D(T)) ≈ 1//2
            @test measure(setup_Triangle3D(T)) ≈ 1/√(T(2))
            @test abs(measure(setup_Quadrilateral2D(T)) - 1) < 1e-6
            @test abs(measure(setup_Quadrilateral3D(T)) - 1.28078927557) < 1e-6
        end
    end

    @testset "QuadraticPolygon" begin
        for T in [Float32, Float64, BigFloat]
            @test abs(measure(setup_QuadraticTriangle2D(T)) - 0.4333333333) < 1e-6
            @test abs(measure(setup_QuadraticTriangle3D(T)) - 0.9058066937) < 1e-6
            @test abs(measure(setup_QuadraticQuadrilateral2D(T)) - 0.86666666) < 1e-6
            @test abs(measure(setup_QuadraticQuadrilateral3D(T)) - 1.39710509) < 1e-6
        end
    end

    # TODO: ConvexPolyhedron, QuadraticPolyhedron
end
