using MOCNeutronTransport
using StaticArrays
@testset "Point_2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Operators" begin
            p₁ = Point_2D{F}(1, 2)
            p₂ = Point_2D{F}(2, 4)

            # Point_2D equivalence
            @test p₁ == Point_2D{F}(1, 2)

            # Point_2D isapprox
            p = Point_2D{F}(1, 2 - 10*eps(F))
            @test F(2) ≈ 2 - 10*eps(F)
            @test p ≈ p₁

            # Point_2D addition
            p = p₁ + p₂
            @test p == [3, 6]

            # Point_2D subtraction
            p = p₁ - p₂
            @test p == [-1, -2]

            # Cross product
            p₁ = Point_2D{F}(2, 3)
            p₂ = Point_2D{F}(5, 6)
            v = p₁ × p₂
            @test v ≈ -3

            # Dot product
            @test p₁ ⋅ p₂ ≈ 10 + 18

            # Number addition
            p₁ = Point_2D{F}(1, 2)
            p₂ = Point_2D{F}(2, 4)
            p = p₁ + 1
            @test p  ≈ [2, 3]

            # Broadcast addition
            parray = [p₁, p₂]
            parray = parray .+ 1
            @test parray[1] == p₁ + 1
            @test parray[2] == p₂ + 1

            # Subtraction
            p = p₁ - 1
            @test p == [0, 1]

            # Multiplication
            p = 4*p₁
            @test p == [4, 8]

            # Division
            p = p₁/4
            @test p == [1//4, 1//2]

            # Unary -
            p = -p₁
            @test p == [-1, -2]

            # Matrix multiplcation
            A = SMatrix{2, 2, F, 4}(1, 2, 3, 4)
            p = Point_2D{F}(1, 2)
            q = A*p
            @test q[1] ≈ 7
            @test q[2] ≈ 10
            @test typeof(q) == Point_2D{F}
        end

        @testset "Methods" begin
            p₁ = Point_2D{F}(1, 2)
            p₂ = Point_2D{F}(2, 4)

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
            p₁ = Point_2D{F}(1, 0)
            p₂ = Point_2D{F}(2, 0)
            p₃ = Point_2D{F}(3, 0)
            points = [p₃, p₁, p₂]
            points_sorted = sortpoints(Point_2D{F}(0, 0), points)
            @test points_sorted[1] == p₁
            @test points_sorted[2] == p₂
            @test points_sorted[3] == p₃
            # mutating
            sortpoints!(Point_2D{F}(0, 0), points)
            @test points[1] == p₁
            @test points[2] == p₂
            @test points[3] == p₃
        end
    end
end
