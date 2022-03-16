@testset "Point" begin
    @testset "Point1D" begin
        for T ∈ Floats
            p₁ = Point(T(1))
            p₂ = Point(T(2))
    
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
            p = Point(T(1)) * Point(T(3))
            @test p == [3]
    
            # Division
            p = p₁/4
            @test p == [1//4]
            p = Point(T(8)) / Point(T(2))
            @test p == [4]
    
            # Dot product
            p₁ = Point(T(2))       
            p₂ = Point(T(5))
            @test p₁ ⋅ p₂ ≈ 10
    
            # ≈
            @test Point(T(2.000001)) ≈ Point(T(2))
            @test Point(T(2.0001)) ≉ Point(T(2))
    
            p₁ = Point(T(-1))
            p₂ = Point(T(4))
    
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
        end
    end
    
    @testset "Point2D" begin
        for T ∈ Floats
            p₁ = Point2D{T}(1, 2)
            p₂ = Point2D{T}(2, 4)
    
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
            p = Point2D{T}(1,2) * Point2D{T}(3,4)
            @test p == [3, 8]
    
            # Division
            p = p₁/4
            @test p == [1//4, 1//2]
            p = Point2D{T}(8,4) / Point2D{T}(2,4)
            @test p == [4, 1]
    
            # Dot product
            p₁ = Point2D{T}(2, 3)       
            p₂ = Point2D{T}(5, 6)
            @test p₁ ⋅ p₂ ≈ 10 + 18
    
            # Cross product
            v = p₁ × p₂
            @test v ≈ -3
    
            # ≈
            @test Point2D{T}(1, 2.000001) ≈ Point2D{T}(1,2)
            @test Point2D{T}(1, 2.0001) ≉ Point2D{T}(1,2)
            
            p₁ = Point2D{T}(1, 2)
            p₂ = Point2D{T}(2, 4)
    
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

            # isCCW
            @test isCCW(Point2D{T}(0,0), Point2D{T}(1,0), Point2D{T}(1,  1))
            @test !isCCW(Point2D{T}(0,0), Point2D{T}(1,0), Point2D{T}(1, -1))
        end
    end
    
    @testset "Point3D" begin
        for T ∈ Floats
            p₁ = Point3D{T}(1, 1, 0)
            p₂ = Point3D{T}(1, 0, 1)
    
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
            p = Point3D{T}(1,2,3) * Point3D{T}(3,4,5)
            @test p == [3, 8, 15]
    
            # Division
            p = p₁/4
            @test p == [1//4, 1//4, 0]
            p = Point3D{T}(8,4,2) / Point3D{T}(2,4,-1)
            @test p == [4, 1, -2]
    
            # Dot product
            p₁ = Point3D{T}(2, 3, 4)       
            p₂ = Point3D{T}(5, 6, 7)
            @test p₁ ⋅ p₂ ≈ 10 + 18 + 28
    
            # Cross product
            v = p₁ × p₂
            @test v == [-3, 6, -3]
    
            # ≈
            @test Point3D{T}(1, 1, 2.000001) ≈ Point3D{T}(1,1,2)
            @test Point3D{T}(1, 1, 2.0001) ≉ Point3D{T}(1,1,2)
    
            p₁ = Point3D{T}(1, 2, 1)
            p₂ = Point3D{T}(2, 4, 0)
    
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
        end
    end
end
