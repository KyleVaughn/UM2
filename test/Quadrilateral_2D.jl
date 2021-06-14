using MOCNeutronTransport
@testset "Quadrilateral_2D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 0, 1)
            quad = Quadrilateral_2D((p₁, p₂, p₃, p₄))
            @test quad.points == (p₁, p₂, p₃, p₄)

            # single constructor
            quad = Quadrilateral_2D(p₁, p₂, p₃, p₄)
            @test quad.points == (p₁, p₂, p₃, p₄)
        end

        @testset "Methods" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 0, 1)
            quad = Quadrilateral_2D((p₁, p₂, p₃, p₄))

            # interpolation
            quad(0, 0) ≈ p₁
            quad(1, 0) ≈ p₂
            quad(0, 1) ≈ p₃
            quad(1, 1) ≈ p₄
            quad(1//2, 1//2) ≈ Point_2D(T, 1//2, 1//2)

            # area
            a = area(quad)
            @test typeof(a) == typeof(T(1))
            @test a == T(1)

            # in
            p = Point_2D(T, 1//2, 1//10)
            @test p ∈  quad
            p = Point_2D(T, 1//2, 0)
            @test p ∈  quad
            p = Point_2D(T, 1//2, -1//10)
            @test p ∉ quad
        end
    end
end
