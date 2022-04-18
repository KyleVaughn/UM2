@testset "Point" begin
    @testset "1D Points" begin
        for T ∈ Points1D
            P₁ = T(1)
            P₂ = T(2)
    
            # Subtraction
            P = P₁ - P₂
            @test P == [-1]
    
            # ≈
            @test isapprox(T(2.00001), T(2); atol=1e-4)
            @test !isapprox(T(2.001), T(2); atol=1e-4)

            # fast_isapprox
            fast_isapprox(T(2.00001), T(2))
            !fast_isapprox(T(2.001), T(2))

            P₁ = T(-1)
            P₂ = T(4)
            # distance
            @test distance(P₁, P₂) ≈ 5
    
            # distance²
            @test distance²(P₁, P₂) ≈ 25
    
            # midPoint
            mp = midpoint(P₁, P₂)
            @test mp[1] ≈ 3//2
        end
    end
    
    @testset "2D Points" begin
        for T ∈ Points2D
            P₁ = T(1, 2)
            P₂ = T(2, 4)
    
            # Subtraction
            P = P₁ - P₂
            @test P == [-1, -2]
    
            # ≈
            @test isapprox(T(1, 2.00001), T(1,2); atol=1e-4)
            @test !isapprox(T(1, 2.001), T(1,2); atol=1e-4)

            # fast_isapprox
            fast_isapprox(T(1, 2.00001), T(1,2))
            !fast_isapprox(T(1, 2.001), T(1,2))
            
            P₁ = T(1, 2)
            P₂ = T(2, 4)
    
            # distance
            @test distance(P₁, P₂) ≈ sqrt(5)
    
            # distance²
            @test distance²(P₁, P₂) ≈ 5
    
            # midpoint
            mp = midpoint(P₁, P₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3

            # isCCW
            @test isCCW(T(0,0), T(1,0), T(1,  1))
            @test !isCCW(T(0,0), T(1,0), T(1, -1))
        end
    end
#    
#    @testset "Point3" begin
#        for T ∈ Floats
#            p₁ = Point3{T}(1, 1, 0)
#            p₂ = Point3{T}(1, 0, 1)
#    
#            # Subtraction
#            p = p₁ - p₂
#            @test p == [0, 1, -1]
#    
#            # ≈
#            @test Point3{T}(1, 1, 2.000001) ≈ Point3{T}(1,1,2)
#            @test Point3{T}(1, 1, 2.0001) ≉ Point3{T}(1,1,2)
#    
#            p₁ = Point3{T}(1, 2, 1)
#            p₂ = Point3{T}(2, 4, 0)
#    
#            # distance
#            @test distance(p₁, p₂) ≈ sqrt(6)
#    
#            # distance²
#            @test distance²(p₁, p₂) ≈ 6
#    
#            # midpoint
#            mp = midpoint(p₁, p₂)
#            @test mp[1] ≈ 3//2
#            @test mp[2] ≈ 3
#            @test mp[3] ≈ 1//2
#        end
#    end
end
