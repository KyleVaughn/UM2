using MOCNeutronTransport
using StaticArrays
@testset "Point1D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Operators" begin
            p₁ = Point(F(1))
            p₂ = Point(F(2))

            # Unary -
            p = -p₁
            @test p == [-1]

            # Addition
            p = p₁ + 1
            @test p == [2]
            p = p₁ + p₂
            @test p == [3]

            # Subtraction
            p = p₁ - 1
            @test p == [0]
            p = p₁ - p₂
            @test p == [-1]

            # Multiplication
            p = 4*p₁
            @test p == [4]
            p = Point(F(1)) * Point(F(3))
            @test p == [3]

            # Division
            p = p₁/4
            @test p == [1//4]
            p = Point(F(8)) / Point(F(2))
            @test p == [4]

            # Dot product
            p₁ = Point(F(2))       
            p₂ = Point(F(5))
            @test p₁ ⋅ p₂ ≈ 10

            # ≈
            @test F(2) ≈ 2 - 10*eps(F)
            @test Point(F(2 - 10*eps(F))) ≈ Point(F(2))
        end

        @testset "Methods" begin
            p₁ = Point(F(-1))
            p₂ = Point(F(4))

            # distance
            @test distance(p₁, p₂) ≈ 5

            # distance²
            @test distance²(p₁, p₂) ≈ 25

            # norm
            @test norm(p₁) ≈ 1
            @test norm(p₂) ≈ 4

            # norm²
            @test norm²(p₁) ≈ 1
            @test norm²(p₂) ≈ 16

            # midpoint
            mp = midpoint(p₁, p₂)
            @test mp[1] ≈ 3//2

            # sort!
            p  = Point(F(0))
            p₁ = Point(F(1))
            p₂ = Point(F(2))
            p₃ = Point(F(3))
            points = [p₃, p₁, p₂]
            sort!(p, points)
            @test points[1] == p₁
            @test points[2] == p₂
            @test points[3] == p₃

            # sort
            p  = Point(F(0))
            p₁ = Point(F(1))
            p₂ = Point(F(2))
            p₃ = Point(F(3))
            points = [p₃, p₁, p₂]
            points_sorted = sort(p, points)
            @test points_sorted[1] == p₁
            @test points_sorted[2] == p₂
            @test points_sorted[3] == p₃
        end
    end
end

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
            p = Point2D{F}(1,2) * Point2D{F}(3,4)
            @test p == [3, 8]

            # Division
            p = p₁/4
            @test p == [1//4, 1//2]
            p = Point2D{F}(8,4) / Point2D{F}(2,4)
            @test p == [4, 1]

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

            # sort!
            p  = Point2D{F}(0, 0)
            p₁ = Point2D{F}(1, 0)
            p₂ = Point2D{F}(2, 0)
            p₃ = Point2D{F}(3, 0)
            points = [p₃, p₁, p₂]
            sort!(p, points)
            @test points[1] == p₁
            @test points[2] == p₂
            @test points[3] == p₃

            # sort
            p  = Point2D{F}(0, 0)
            p₁ = Point2D{F}(1, 0)
            p₂ = Point2D{F}(2, 0)
            p₃ = Point2D{F}(3, 0)
            points = [p₃, p₁, p₂]
            points_sorted = sort(p, points)
            @test points_sorted[1] == p₁
            @test points_sorted[2] == p₂
            @test points_sorted[3] == p₃
        end
    end
end

@testset "Point3D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Operators" begin
            p₁ = Point3D{F}(1, 1, 0)
            p₂ = Point3D{F}(1, 0, 1)

            # Unary -
            p = -p₁
            @test p == [-1, -1, 0]

            # Addition
            p = p₁ + 1
            @test p == [2, 2, 1]
            p = p₁ + p₂
            @test p == [2, 1, 1]

            # Subtraction
            p = p₁ - 1
            @test p == [0, 0, -1]
            p = p₁ - p₂
            @test p == [0, 1, -1]

            # Multiplication
            p = 4*p₁
            @test p == [4, 4, 0]
            p = Point3D{F}(1,2,3) * Point3D{F}(3,4,5)
            @test p == [3, 8, 15]

            # Division
            p = p₁/4
            @test p == [1//4, 1//4, 0]
            p = Point3D{F}(8,4,2) / Point3D{F}(2,4,-1)
            @test p == [4, 1, -2]

            # Dot product
            p₁ = Point3D{F}(2, 3, 4)       
            p₂ = Point3D{F}(5, 6, 7)
            @test p₁ ⋅ p₂ ≈ 10 + 18 + 28

            # Cross product
            v = p₁ × p₂
            @test v == [-3, 6, -3]

            # ≈
            @test F(2) ≈ 2 - 10*eps(F)
            @test Point3D{F}(1, 2 - 10*eps(F), 1) ≈ Point3D{F}(1, 2, 1)
        end

        @testset "Methods" begin
            p₁ = Point3D{F}(1, 2, 1)
            p₂ = Point3D{F}(2, 4, 0)

            # distance
            @test distance(p₁, p₂) ≈ sqrt(6)

            # distance²
            @test distance²(p₁, p₂) ≈ 6

            # norm
            @test norm(p₁) ≈ sqrt(6)
            @test norm(p₂) ≈ sqrt(4 + 16)

            # norm²
            @test norm²(p₁) ≈ 6
            @test norm²(p₂) ≈ 4 + 16

            # midpoint
            mp = midpoint(p₁, p₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3
            @test mp[3] ≈ 1//2

            # sort!
            p  = Point3D{F}(0, 0, 0)
            p₁ = Point3D{F}(1, 0, 0)
            p₂ = Point3D{F}(2, 0, 0)
            p₃ = Point3D{F}(3, 0, 0)
            points = [p₃, p₁, p₂]
            sort!(p, points)
            @test points[1] == p₁
            @test points[2] == p₂
            @test points[3] == p₃

            # sort
            p  = Point3D{F}(0, 0, 0)
            p₁ = Point3D{F}(1, 0, 0)
            p₂ = Point3D{F}(2, 0, 0)
            p₃ = Point3D{F}(3, 0, 0)
            points = [p₃, p₁, p₂]
            points_sorted = sort(p, points)
            @test points_sorted[1] == p₁
            @test points_sorted[2] == p₂
            @test points_sorted[3] == p₃
        end
    end
end
