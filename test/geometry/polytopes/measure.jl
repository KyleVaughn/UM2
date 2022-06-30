@testset "Measure" begin
    @testset "AABox" begin for T in Floats
        @test measure(setup_AABox2(T)) ≈ 1
        @test measure(setup_AABox3(T)) ≈ 1
        if T != BigFloat
            aab = setup_AABox2(T)
            @test @ballocated(measure($aab), samples = 1, evals = 2) == 0
        end
    end end

    @testset "LineSegment" begin for T in Floats
        @test measure(setup_LineSegment2(T)) ≈ sqrt(T(2))
        @test measure(setup_LineSegment3(T)) ≈ sqrt(T(3))
        if T != BigFloat
            l = setup_LineSegment2(T)
            @test @ballocated(measure($l), samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticSegment" begin for T in Floats
        q₂ = QuadraticSegment(Point{2, T}(0, 0), Point{2, T}(1, 0), Point{2, T}(1 // 2, 0))
        @test measure(q₂) ≈ 1
        q₂ = setup_QuadraticSegment2(T)
        @test abs(measure(q₂) - 1.4789428575445974) < 1e-6
        q₃ = QuadraticSegment(Point{3, T}(0, 0, 0),
                              Point{3, T}(1, 0, 0),
                              Point{3, T}(1 // 2, 0, 0))
        @test measure(q₃) ≈ 1
        q₃ = setup_QuadraticSegment3(T)
        @test abs(measure(q₃) - 1.7978527887818835) < 1e-6
        if T != BigFloat
            q = setup_QuadraticSegment2(T)
            @test @ballocated(measure($q), samples = 1, evals = 2) == 0
        end
    end end

    @testset "Polygon" begin for T in Floats
        @test measure(setup_Triangle2(T)) ≈ 1 // 2
        @test measure(setup_Triangle3(T)) ≈ 1 / √(T(2))
        @test abs(measure(setup_Quadrilateral2(T)) - 1) < 1e-6
        @test abs(measure(setup_Quadrilateral3(T)) - 1.28078927557) < 1e-6
        if T != BigFloat
            tri = setup_Triangle2(T)
            quad = setup_Quadrilateral2(T)
            @test @ballocated(measure($tri), samples = 1, evals = 2) == 0
            @test @ballocated(measure($quad), samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticPolygon" begin for T in Floats
        @test abs(measure(setup_QuadraticTriangle2(T)) - 0.4333333333) < 1e-6
        @test abs(measure(setup_QuadraticTriangle3(T)) - 0.9058066937) < 1e-6
        @test abs(measure(setup_QuadraticQuadrilateral2(T)) - 0.86666666) < 1e-6
        @test abs(measure(setup_QuadraticQuadrilateral3(T)) - 1.39710509) < 1e-6
        if T != BigFloat
            tri6 = setup_QuadraticTriangle2(T)
            quad8 = setup_QuadraticQuadrilateral2(T)
            @test @ballocated(measure($tri6), samples = 1, evals = 2) == 0
            @test @ballocated(measure($quad8), samples = 1, evals = 2) == 0
        end
    end end
end
