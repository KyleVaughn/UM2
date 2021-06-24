using MOCNeutronTransport
@testset "Quadrilateral8_2D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 0, 1)
            p₅ = Point_2D(T, 1//2,    0)
            p₆ = Point_2D(T,    1, 1//2)
            p₇ = Point_2D(T, 1//2,    1)
            p₈ = Point_2D(T,    0, 1//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # single constructor
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end

        @testset "Methods" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 0, 1)
            p₅ = Point_2D(T, 1//2,    0)
            p₆ = Point_2D(T,    1, 1//2)
            p₇ = Point_2D(T, 1//2,    1)
            p₈ = Point_2D(T,    0, 1//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            # interpolation
            @test quad8(0, 0) ≈ p₁
            @test quad8(1, 0) ≈ p₂
            @test quad8(1, 1) ≈ p₃
            @test quad8(0, 1) ≈ p₄
            @test quad8(1//2,    0) ≈ p₅
            @test quad8(   1, 1//2) ≈ p₆
            @test quad8(1//2,    1) ≈ p₇
            @test quad8(   0, 1//2) ≈ p₈
            @test quad8(1//2, 1//2) ≈ Point_2D(T, 1//2, 1//2)

            # area
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 3)
            p₄ = Point_2D(T, 0, 3)
            p₅ = Point_2D(T, 3//2, 1//2)
            p₆ = Point_2D(T, 5//2, 3//2)
            p₇ = Point_2D(T, 3//2, 5//2)
            p₈ = Point_2D(T, 0,    3//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            # 2D default
            @test isapprox(area(quad8; N = 4), 17//3, atol=1.0e-6)
            # 3D default
            p₈ = Point_2D(T, 0, 3//2, 2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test isapprox(area(quad8; N = 15), 10.09431, atol=1.0e-4)

            # intersect
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 3)
            p₄ = Point_2D(T, 0, 3)
            p₅ = Point_2D(T, 3//2, 1//2)
            p₆ = Point_2D(T, 5//2, 3//2)
            p₇ = Point_2D(T, 3//2, 5//2)
            p₈ = Point_2D(T, 0,    3//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment_3D(Point_2D(T, 1, 3//2, -2),
                               Point_2D(T, 1, 3//2,  2))
            intersection = l ∩ quad8

            # 1 intersection
            @test intersection[1]
            @test intersection[2] == 1
            @test intersection[3] ≈ Point_2D(T, 1, 3//2, 0)

            # 0 intersection
            l = LineSegment_3D(Point_2D(T, 1, 0, -2),
                               Point_2D(T, 1, 0,  2))
            intersection = l ∩ quad8
            @test !intersection[1]

            # 2 intersections
            p₈ = Point_2D(T, 0, 3//2, 2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment_3D(Point_2D(T, 1, 0, 1//2),
                               Point_2D(T, 1, 3, 1//2))
            intersection = l ∩ quad8
            @test intersection[1]
            @test intersection[2] == 2
            @test norm(intersection[3] - Point_2D(T, 1, 0.61244509, 1//2)) < 1.0e-6
            @test norm(intersection[4] - Point_2D(T, 1, 2.38737343, 1//2)) < 1.0e-6
        end
    end
end
