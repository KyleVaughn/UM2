@testset "Point" begin
    @testset "Point{1}" begin
        for T ∈ Floats 
            P₁ = Point{1,T}(1)
            P₂ = Point{1,T}(2)
    
            # Subtraction
            P = P₁ - P₂
            @test P == [-1]
    
            # ≈
            @test Point{1,T}(2 + 1e-5) ≈ Point{1,T}(2)
            @test Point{1,T}(2 + 1e-3) ≉ Point{1,T}(2)

            P₁ = Point{1,T}(-1)
            P₂ = Point{1,T}(4)
            # distance
            @test distance(P₁, P₂) ≈ 5
    
            # distance²
            @test distance²(P₁, P₂) ≈ 25
    
            # midPoint
            mp = midpoint(P₁, P₂)
            @test mp[1] ≈ 3//2
        end
    end
    
    @testset "Point{2}" begin
        for T ∈ Floats 
            P₁ = Point{2,T}(1, 2)
            P₂ = Point{2,T}(2, 4)
    
            # Subtraction
            P = P₁ - P₂
            @test P == [-1, -2]
    
            # ≈
            @test Point{2,T}(1, 2 + 1e-5) ≈ Point{2,T}(1,2)
            @test Point{2,T}(1, 2 + 1e-3) ≉ Point{2,T}(1,2)
            
            P₁ = Point{2,T}(1, 2)
            P₂ = Point{2,T}(2, 4)
    
            # distance
            @test distance(P₁, P₂) ≈ sqrt(5)
    
            # distance²
            @test distance²(P₁, P₂) ≈ 5
    
            # midpoint
            mp = midpoint(P₁, P₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3

            # isCCW
            @test  isCCW(Point{2,T}(0,0), Point{2,T}(1,0), Point{2,T}(1,  1))
            @test !isCCW(Point{2,T}(0,0), Point{2,T}(1,0), Point{2,T}(1, -1))
        end
    end
    
    @testset "Point{3}" begin
        for T ∈ Floats 
            P₁ = Point{3,T}(1, 1, 0)
            P₂ = Point{3,T}(1, 0, 1)
    
            # Subtraction
            P = P₁ - P₂
            @test P == [0, 1, -1]
    
            # ≈
            @test Point{3,T}(1, 1, 2 + 1e-5) ≈ Point{3,T}(1,1,2)
            @test Point{3,T}(1, 1, 2 + 1e-3) ≉ Point{3,T}(1,1,2)
    
            P₁ = Point{3,T}(1, 2, 1)
            P₂ = Point{3,T}(2, 4, 0)
    
            # distance
            @test distance(P₁, P₂) ≈ sqrt(6)
    
            # distance²
            @test distance²(P₁, P₂) ≈ 6
    
            # midpoint
            mp = midpoint(P₁, P₂)
            @test mp[1] ≈ 3//2
            @test mp[2] ≈ 3
            @test mp[3] ≈ 1//2
        end
    end
end
