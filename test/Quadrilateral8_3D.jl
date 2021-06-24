using MOCNeutronTransport
@testset "Quadrilateral8_3D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 0, 1)
            p₅ = Point_3D(T, 1//2,    0)
            p₆ = Point_3D(T,    1, 1//2)
            p₇ = Point_3D(T, 1//2,    1)
            p₈ = Point_3D(T,    0, 1//2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)

            # single constructor
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8.points == (p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end

        @testset "Methods" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 0, 1)
            p₅ = Point_3D(T, 1//2,    0)
            p₆ = Point_3D(T,    1, 1//2)
            p₇ = Point_3D(T, 1//2,    1)
            p₈ = Point_3D(T,    0, 1//2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            # interpolation
            @test quad8(0, 0) ≈ p₁
            @test quad8(1, 0) ≈ p₂
            @test quad8(1, 1) ≈ p₃
            @test quad8(0, 1) ≈ p₄
            @test quad8(1//2,    0) ≈ p₅
            @test quad8(   1, 1//2) ≈ p₆
            @test quad8(1//2,    1) ≈ p₇
            @test quad8(   0, 1//2) ≈ p₈
            @test quad8(1//2, 1//2) ≈ Point_3D(T, 1//2, 1//2)

            # area
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 2)
            p₃ = Point_3D(T, 2, 3)
            p₄ = Point_3D(T, 0, 3)
            p₅ = Point_3D(T, 3//2, 1//2)
            p₆ = Point_3D(T, 5//2, 3//2)
            p₇ = Point_3D(T, 3//2, 5//2)
            p₈ = Point_3D(T, 0,    3//2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

            p₈ = Point_3D(T, 0, 3//2, 2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test isapprox(area(quad8), 10.09431, atol=1.0e-4)

            # intersect
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 2)
            p₃ = Point_3D(T, 2, 3)
            p₄ = Point_3D(T, 0, 3)
            p₅ = Point_3D(T, 3//2, 1//2)
            p₆ = Point_3D(T, 5//2, 3//2)
            p₇ = Point_3D(T, 3//2, 5//2)
            p₈ = Point_3D(T, 0,    3//2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment_3D(Point_3D(T, 1, 3//2, -2),
                               Point_3D(T, 1, 3//2,  2))
            intersection = l ∩ quad8

            # 1 intersection
            @test intersection[1]
            @test intersection[2] == 1
            @test intersection[3] ≈ Point_3D(T, 1, 3//2, 0)

            # 0 intersection
            l = LineSegment_3D(Point_3D(T, 1, 0, -2),
                               Point_3D(T, 1, 0,  2))
            intersection = l ∩ quad8
            @test !intersection[1]

            # 2 intersections
            p₈ = Point_3D(T, 0, 3//2, 2)
            quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            l = LineSegment_3D(Point_3D(T, 1, 0, 1//2),
                               Point_3D(T, 1, 3, 1//2))
            intersection = l ∩ quad8
            @test intersection[1]
            @test intersection[2] == 2
            @test norm(intersection[3] - Point_3D(T, 1, 0.6099032181207209, 1//2)) < 1.0e-3
            @test norm(intersection[4] - Point_3D(T, 1, 2.390096778327549, 1//2)) < 1.0e-3
        end
    end
end
