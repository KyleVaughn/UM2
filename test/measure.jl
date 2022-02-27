@testset "Measure" begin
    @testset "LineSegment" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(unit_LineSegment2D(T)) ≈ sqrt(T(2))
            @test measure(unit_LineSegment3D(T)) ≈ sqrt(T(3))
        end
    end

    @testset "QuadraticSegment" begin
        for T in [Float32, Float64, BigFloat]
            q₂ = QuadraticSegment(Point2D{T}(0,0), 
                                  Point2D{T}(1,0), 
                                  Point2D{T}(1//2, 0))
            @test measure(q₂) ≈ 1
            q₂ = unit_QuadraticSegment2D(T)
            @test abs(measure(q₂) - 1.4789428575445974) < 1e-6
            q₃ = QuadraticSegment(Point3D{T}(0,0,0), 
                                  Point3D{T}(1,0,0), 
                                  Point3D{T}(1//2,0,0))
            @test measure(q₃) ≈ 1
            q₃ = unit_QuadraticSegment3D(T)
            @test abs(measure(q₃) - 1.7978527887818835) < 1e-6
        end
    end

    @testset "AABox" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(unit_AABox2D(T)) ≈ 1
            @test measure(unit_AABox3D(T)) ≈ 1
        end
    end

    @testset "ConvexPolygon" begin
        for T in [Float32, Float64, BigFloat]
            @test measure(unit_Triangle2D(T)) ≈ 1//2
            @test measure(unit_Triangle3D(T)) ≈ 1/√(T(2))
            @test abs(measure(unit_Quadrilateral2D(T)) - 1) < 1e-6
            @test abs(measure(unit_Quadrilateral3D(T)) - 1.28078927557) < 1e-6
        end
    end

    @testset "QuadraticPolygon" begin
        for T in [Float32, Float64, BigFloat]
            @test abs(measure(unit_QuadraticTriangle2D(T)) - 0.4333333333) < 1e-6
            @test abs(measure(unit_QuadraticTriangle3D(T)) - 1//2) < 1e-6
#            @test abs(measure(unit_Quadrilateral2D(T)) - 1) < 1e-6
#            @test abs(measure(unit_Quadrilateral3D(T)) - 1) < 1e-6
        end
    end

#            # area
#            p₀ = Point2D{T}(1,1)
#            p₁ = Point2D(Point2D{T}(0, 0) + p₀) 
#            p₂ = Point2D(Point2D{T}(2, 0) + p₀) 
#            p₃ = Point2D(Point2D{T}(2, 2) + p₀) 
#            p₄ = Point2D(Point2D{T}(3//2, 1//4) + p₀) 
#            p₅ = Point2D(Point2D{T}(3, 1) + p₀) 
#            p₆ = Point2D(Point2D{T}(1, 1) + p₀) 
#            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆) 
#            @test isapprox(area(tri6), 3, atol=1.0e-6)
#
#                # area
#        p₁ = Point2D{T}(0, 0)
#        p₂ = Point2D{T}(2, 0)
#        p₃ = Point2D{T}(2, 3)
#        p₄ = Point2D{T}(0, 3)
#        p₅ = Point2D{T}(3//2, 1//2)
#        p₆ = Point2D{T}(5//2, 3//2)
#        p₇ = Point2D{T}(3//2, 5//2)
#        p₈ = Point2D{T}(0,    3//2)
#        quad8 = QuadraticQuadrilateral(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
#        @test isapprox(area(quad8), 17//3, atol=1.0e-6)
end
