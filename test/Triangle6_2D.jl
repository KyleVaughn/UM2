using MOCNeutronTransport
@testset "Triangle6_2D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 1//2)
            p₅ = Point_2D(T, 1, 1//2)
            p₆ = Point_2D(T, 1//2, 1//2)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)

            # single constructor
            tri6 = Triangle6_2D(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)
        end

        @testset "Methods" begin
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 1)
            p₃ = Point_2D(T, 1, 1)
            p₄ = Point_2D(T, 1//2)
            p₅ = Point_2D(T, 1, 1//2)
            p₆ = Point_2D(T, 1//2, 1//2)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))

            # interpolation
            tri6(0, 0) ≈ p₁
            tri6(1, 0) ≈ p₂
            tri6(0, 1) ≈ p₃
            tri6(1//2, 0) ≈ p₄
            tri6(1//2, 1//2) ≈ p₅
            tri6(0, 1//2) ≈ p₆
            tri6(1//2, 1//2) ≈ Point_2D(T, 1//2, 1//2)

            # area
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 2)
            p₄ = Point_2D(T, 1, 1//4)
            p₅ = Point_2D(T, 3, 1)
            p₆ = Point_2D(T, 1, 1)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            # 2D default
            @test isapprox(area(tri6; N = 12), 3, atol=1.0e-6)

            # in
            p₁ = Point_2D(T, 0)
            p₂ = Point_2D(T, 2)
            p₃ = Point_2D(T, 2, 2)
            p₄ = Point_2D(T, 1, 1//4)
            p₅ = Point_2D(T, 3, 1)
            p₆ = Point_2D(T, 1, 1)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test Point_2D(T, 1, 1//2) ∈ tri6
            @test Point_2D(T, 1, 0) ∉  tri6
        end
    end
end
