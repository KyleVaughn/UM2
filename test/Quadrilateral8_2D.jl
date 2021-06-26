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
            @test isapprox(area(quad8; N = 3), 17//3, atol=1.0e-6)

            # real_to_parametric
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 3)
            p₄ = Point_2D(T, 0, 3)
            p₅ = Point_2D(T, 3//2, 1//2)
            p₆ = Point_2D(T, 5//2, 3//2)
            p₇ = Point_2D(T, 3//2, 5//2)
            p₈ = Point_2D(T, 0, 3//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test quad8(real_to_parametric(p₁, quad8)) ≈ p₁
            @test quad8(real_to_parametric(p₂, quad8)) ≈ p₂
            @test quad8(real_to_parametric(p₃, quad8)) ≈ p₃
            @test quad8(real_to_parametric(p₄, quad8)) ≈ p₄
            @test quad8(real_to_parametric(p₅, quad8)) ≈ p₅
            @test quad8(real_to_parametric(p₆, quad8)) ≈ p₆

            # in
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 3)
            p₄ = Point_2D(T, 0, 3)
            p₅ = Point_2D(T, 3//2, 1//2)
            p₆ = Point_2D(T, 5//2, 3//2)
            p₇ = Point_2D(T, 3//2, 5//2)
            p₈ = Point_2D(T, 0, 3//2)
            quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
            @test Point_2D(T, 1, 1) ∈  quad8 
            @test Point_2D(T, 1, 0) ∉  quad8 
        end
    end
end
