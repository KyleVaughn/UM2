using MOCNeutronTransport
@testset "Quadrilateral_2D" begin
    for F in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            p₄ = Point_2D(F, 0, 1)
            quad = Quadrilateral_2D(SVector(p₁, p₂, p₃, p₄))
            @test quad.points == SVector(p₁, p₂, p₃, p₄)

            # single constructor
            quad = Quadrilateral_2D(p₁, p₂, p₃, p₄)
            @test quad.points == SVector(p₁, p₂, p₃, p₄)
        end

        @testset "Methods" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            p₄ = Point_2D(F, 0, 1)
            quad = Quadrilateral_2D(SVector(p₁, p₂, p₃, p₄))

            # interpolation
            @test quad(0, 0) ≈ p₁
            @test quad(1, 0) ≈ p₂
            @test quad(1, 1) ≈ p₃
            @test quad(0, 1) ≈ p₄
            @test quad(1//2, 1//2) ≈ Point_2D(F, 1//2, 1//2)

            # area
            a = area(quad)
            @test typeof(a) == F
            @test a == F(1)

            # in
            p = Point_2D(F, 1//2, 1//10)
            @test p ∈  quad
            p = Point_2D(F, 1//2, -1//10)
            @test p ∉ quad

            # 3 intersections
            l = LineSegment_2D(p₃, p₁)
            ipoints, points = intersect(l, quad)
            @test ipoints == 3
            @test points[1] ≈ p₁
            @test points[2] ≈ p₃
            @test points[3] ≈ p₃

            # 2 intersections
            l = LineSegment_2D(Point_2D(F, 0, 1//2), Point_2D(F, 1, 1//2))
            ipoints, points = intersect(l, quad)
            @test ipoints == 2
            @test points[1] ≈ Point_2D(F, 1, 1//2)
            @test points[2] ≈ Point_2D(F, 0, 1//2)

            # 0 intersections
            l = LineSegment_2D(Point_2D(F, -1, -1), Point_2D(F, 2, -1))
            ipoints, points = intersect(l, quad)
            @test ipoints == 0
        end
    end
end
