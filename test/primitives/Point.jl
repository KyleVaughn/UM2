using MOCNeutronTransport
using StaticArrays
@testset "Point2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Operators" begin
            p₁ = Point2D{F}(1, 2)
            p₂ = Point2D{F}(2, 4)

            # Unary -
            p = -p₁
            @test p == [-1, -2]

            # Addition
            p = p₁ + 1
            @test p == [2, 3]
            p = p₁ + p₂
            @test p == [3, 6]

            # Subtraction
            p = p₁ - 1
            @test p == [0, 1]
            p = p₁ - p₂
            @test p == [-1, -2]

            # Multiplication
            p = 4*p₁
            @test p == [4, 8]

            # Division
            p = p₁/4
            @test p == [1//4, 1//2]

            # Dot product
            p₁ = Point2D{F}(2, 3)       
            p₂ = Point2D{F}(5, 6)
            @test p₁ ⋅ p₂ ≈ 10 + 18

            # Cross product
            v = p₁ × p₂
            @test v ≈ -3

            # ≈
            @test F(2) ≈ 2 - 10*eps(F)
            @test Point2D{F}(1, 2 - 10*eps(F)) ≈ Point2D{F}(1, 2)
        end

        @testset "Methods" begin
            p₁ = Point2D{F}(1, 2)
            p₂ = Point2D{F}(2, 4)

            # distance
            @test distance(p₁, p₂) ≈ sqrt(5)

            # distance²
            @test distance²(p₁, p₂) ≈ 5

            # norm
            @test norm(p₁) ≈ sqrt(5)
            @test norm(p₂) ≈ sqrt(4 + 16)

            # norm²
            @test norm²(p₁) ≈ 5
            @test norm²(p₂) ≈ 4 + 16

            # midpoint
            mp = midpoint(p₁, p₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3

            # sort_points
            p₁ = Point2D{F}(1, 0)
            p₂ = Point2D{F}(2, 0)
            p₃ = Point2D{F}(3, 0)
            points = [p₃, p₁, p₂]
            sortpoints!(Point2D{F}(0, 0), points)
            @test points[1] == p₁
            @test points[2] == p₂
            @test points[3] == p₃
        end
    end
end
