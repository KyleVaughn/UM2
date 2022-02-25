
@testset "QuadraticTriangle2D" begin
    for T in [Float32, Float64, BigFloat]
        p₁ = Point2D{T}(0, 0)
        p₂ = Point2D{T}(1, 0)
        p₃ = Point2D{T}(1, 1)
        p₄ = Point2D{T}(1//2, 0)
        p₅ = Point2D{T}(1, 1//2)
        p₆ = Point2D{T}(1//2, 1//2)
        tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
        @test tri6.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆)

        # interpolation
        @test tri6(0, 0) ≈ p₁
        @test tri6(1, 0) ≈ p₂
        @test tri6(0, 1) ≈ p₃
        @test tri6(1//2, 0) ≈ p₄
        @test tri6(1//2, 1//2) ≈ p₅
        @test tri6(0, 1//2) ≈ p₆

        # jacobian
        J = jacobian(tri6, 0, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 1, 0)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 0, 1)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1
        J = jacobian(tri6, 1//2, 1//2)
        @test J[1] ≈ 1
        @test abs(J[2]) < 1e-6
        @test abs(J[3]) < 1e-6
        @test J[4] ≈ 1

        # real_to_parametric
        p₁ = Point2D{T}(0, 0)
        p₂ = Point2D{T}(2, 0)
        p₃ = Point2D{T}(2, 2)
        p₄ = Point2D{T}(3//2, 1//4)
        p₅ = Point2D{T}(3, 1)
        p₆ = Point2D{T}(1, 1)
        tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
        @test tri6(real_to_parametric(p₁, tri6)) ≈ p₁
        @test tri6(real_to_parametric(p₂, tri6)) ≈ p₂
        @test tri6(real_to_parametric(p₃, tri6)) ≈ p₃
        @test tri6(real_to_parametric(p₄, tri6)) ≈ p₄
        @test tri6(real_to_parametric(p₅, tri6)) ≈ p₅
        @test tri6(real_to_parametric(p₆, tri6)) ≈ p₆
    end
end
