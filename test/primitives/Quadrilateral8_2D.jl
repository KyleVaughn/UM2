using MOCNeutronTransport
@testset "Quadrilateral8_2D" begin
    for F in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            p₄ = Point_2D(F, 0, 1)
            p₅ = Point_2D(F, 1//2,    0)
            p₆ = Point_2D(F,    1, 1//2)
            p₇ = Point_2D(F, 1//2,    1)
            p₈ = Point_2D(F,    0, 1//2)
            quad8 = Quadrilateral8_2D(SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # single constructor
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            @test quad8.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end

        @testset "Methods" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            p₄ = Point_2D(F, 0, 1)
            p₅ = Point_2D(F, 1//2,    0)
            p₆ = Point_2D(F,    1, 1//2)
            p₇ = Point_2D(F, 1//2,    1)
            p₈ = Point_2D(F,    0, 1//2)
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # interpolation
            @test quad8(0, 0) ≈ p₁
            @test quad8(1, 0) ≈ p₂
            @test quad8(1, 1) ≈ p₃
            @test quad8(0, 1) ≈ p₄
            @test quad8(1//2,    0) ≈ p₅
            @test quad8(   1, 1//2) ≈ p₆
            @test quad8(1//2,    1) ≈ p₇
            @test quad8(   0, 1//2) ≈ p₈
            @test quad8(1//2, 1//2) ≈ Point_2D(F, 1//2, 1//2)

            # derivative
            dr, ds = derivative(quad8, 0, 0)
            @test dr == Point_2D(F, 1, 0)
            @test ds == Point_2D(F, 0, 1)
            dr, ds = derivative(quad8, 1, 0)
            @test dr == Point_2D(F, 1, 0)
            @test ds == Point_2D(F, 0, 1)
            dr, ds = derivative(quad8, 1, 1)
            @test dr == Point_2D(F, 1, 0)
            @test ds == Point_2D(F, 0, 1)

            # area
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 3)
            p₄ = Point_2D(F, 0, 3)
            p₅ = Point_2D(F, 3//2, 1//2)
            p₆ = Point_2D(F, 5//2, 3//2)
            p₇ = Point_2D(F, 3//2, 5//2)
            p₈ = Point_2D(F, 0,    3//2)
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # 2D default
            @test isapprox(area(quad8), 17//3, atol=1.0e-6)

            # real_to_parametric
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 3)
            p₄ = Point_2D(F, 0, 3)
            p₅ = Point_2D(F, 3//2, 1//2)
            p₆ = Point_2D(F, 5//2, 3//2)
            p₇ = Point_2D(F, 3//2, 5//2)
            p₈ = Point_2D(F, 0, 3//2)
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            @test quad8(real_to_parametric(p₁, quad8)) ≈ p₁
            @test quad8(real_to_parametric(p₂, quad8)) ≈ p₂
            @test quad8(real_to_parametric(p₃, quad8)) ≈ p₃
            @test quad8(real_to_parametric(p₄, quad8)) ≈ p₄
            @test quad8(real_to_parametric(p₅, quad8)) ≈ p₅
            @test quad8(real_to_parametric(p₆, quad8)) ≈ p₆

            # in
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 3)
            p₄ = Point_2D(F, 0, 3)
            p₅ = Point_2D(F, 3//2, 1//2)
            p₆ = Point_2D(F, 5//2, 3//2)
            p₇ = Point_2D(F, 3//2, 5//2)
            p₈ = Point_2D(F, 0, 1)
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            @test Point_2D(F, 1, 1) ∈  quad8
            @test Point_2D(F, 1, 0) ∉  quad8

            # intersect
            # 0 intersections
            l = LineSegment_2D(Point_2D(F, 0, -1), Point_2D(F, 4, -1))
            npoints, points = l ∩ quad8
            @test npoints == 0

            # 2 intersections
            l = LineSegment_2D(Point_2D(F, -1, 1), Point_2D(F, 4, 1))
            npoints, points = l ∩ quad8
            @test npoints == 2
            @test points[1] ≈ Point_2D(F, 2.444444, 1)
            @test points[2] ≈ Point_2D(F, 0, 1)

            # 4 intersections
            l = LineSegment_2D(Point_2D(F, 0, 1//10), Point_2D(F, 4, 1//10))
            npoints, points = l ∩ quad8
            @test npoints == 4
            @test points[1] ≈ Point_2D(F, 0.20557280900008415,       1//10)
            @test points[2] ≈ Point_2D(F, 1.9944271909999158,        1//10)
            @test points[3] ≈ Point_2D(F, 2.0644444444444447,        1//10)
            @test points[4] ≈ Point_2D(F, 0,                         1//10)

            # 6 intersections
            p₁ = Point_2D(F,  1, 0)
            p₂ = Point_2D(F,  0, 0)
            p₃ = Point_2D(F, -1, 0)
            p₄ = Point_2D(F,  0, -2)
            p₅ = Point_2D(F,  1//2, -1//2)
            p₆ = Point_2D(F, -1//2, -1//2)
            p₇ = Point_2D(F, -1//2, -3//2)
            p₈ = Point_2D(F,  1//2, -3//2)
            quad8 = Quadrilateral8_2D(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            l = LineSegment_2D(Point_2D(F, -2, -1//4), Point_2D(F, 2, -1//4))
            n, points = l ∩ quad8
            @test n == 6
            @test points[1] ≈ Point_2D{F}(F[ 0.14644659, -1//4])
            @test points[2] ≈ Point_2D{F}(F[ 0.8535534,  -1//4])
            @test points[3] ≈ Point_2D{F}(F[-0.8535534,  -1//4])
            @test points[4] ≈ Point_2D{F}(F[-0.14644665, -1//4])
            @test points[5] ≈ Point_2D{F}(F[-0.9354143,  -1//4])
            @test points[6] ≈ Point_2D{F}(F[ 0.9354143,  -1//4])
        end
    end
end
