@testset "Point" begin
    @testset "Point1D" begin
        for T ∈ Floats
            p₁ = Point(T(1))
            p₂ = Point(T(2))
    
            # Subtraction
            p = p₁ - p₂
            @test p == [-1]
    
            # ≈
            @test Point(T(2.000001)) ≈ Point(T(2))
            @test Point(T(2.0001)) ≉ Point(T(2))
    
            p₁ = Point(T(-1))
            p₂ = Point(T(4))
    
            # distance
            @test distance(p₁, p₂) ≈ 5
    
            # distance²
            @test distance²(p₁, p₂) ≈ 25
    
            # midpoint
            mp = midpoint(p₁, p₂)
            @test mp[1] ≈ 3//2
        end
    end
    
    @testset "Point2D" begin
        for T ∈ Floats
            p₁ = Point2D{T}(1, 2)
            p₂ = Point2D{T}(2, 4)
    
            # Subtraction
            p = p₁ - p₂
            @test p == [-1, -2]
    
            # ≈
            @test Point2D{T}(1, 2.000001) ≈ Point2D{T}(1,2)
            @test Point2D{T}(1, 2.0001) ≉ Point2D{T}(1,2)
            
            p₁ = Point2D{T}(1, 2)
            p₂ = Point2D{T}(2, 4)
    
            # distance
            @test distance(p₁, p₂) ≈ sqrt(5)
    
            # distance²
            @test distance²(p₁, p₂) ≈ 5
    
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
    
            # Subtraction
            p = p₁ - p₂
            @test p == [0, 1, -1]
    
            # ≈
            @test Point3D{T}(1, 1, 2.000001) ≈ Point3D{T}(1,1,2)
            @test Point3D{T}(1, 1, 2.0001) ≉ Point3D{T}(1,1,2)
    
            p₁ = Point3D{T}(1, 2, 1)
            p₂ = Point3D{T}(2, 4, 0)
    
            # distance
            @test distance(p₁, p₂) ≈ sqrt(6)
    
            # distance²
            @test distance²(p₁, p₂) ≈ 6
    
            # midpoint
            mp = midpoint(p₁, p₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3
            @test mp[3] ≈ 1//2
        end
    end
end
