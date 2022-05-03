@testset "Point" begin
    @testset "Point{1}" begin
        for T ∈ Floats 
            P₁ = Point{1,T}(1)
            P₂ = Point{1,T}(2)
            𝘃₂ = Vec{1,T}(2)

            # coordinates
            coords = coordinates(P₁)
            @test coords == [1]
    
            # subtraction
            P = P₁ - P₂
            @test P == [-1]

            # vector addition
            v = P₁ + 𝘃₂
            @test v == [3]

            # vector subtraction 
            v = P₁ - 𝘃₂
            @test v == [-1]
    
            # ≈
            @test Point{1,T}(2 + 0.1*ϵ_Point) ≈ Point{1,T}(2)
            @test Point{1,T}(2 + 10*ϵ_Point) ≉ Point{1,T}(2)

            P₁ = Point{1,T}(-1)
            P₂ = Point{1,T}(4)
            # distance
            @test distance(P₁, P₂) ≈ 5
    
            # distance²
            @test distance²(P₁, P₂) ≈ 25
    
            # midPoint
            mp = midpoint(P₁, P₂)
            @test mp ≈ [3//2]
        end
    end
    
    @testset "Point{2}" begin
        for T ∈ Floats 
            P₁ = Point{2,T}(1, 2)
            P₂ = Point{2,T}(2, 4)
            𝘃₂ = Vec{2,T}(2,4)

            # coordinates
            coords = coordinates(P₁)
            @test coords == [1, 2]
    
            # subtraction
            P = P₁ - P₂
            @test P == [-1, -2]

            # vector addition
            v = P₁ + 𝘃₂
            @test v == [3, 6]

            # vector subtraction 
            v = P₁ - 𝘃₂
            @test v == [-1, -2]
    
            # ≈
            @test Point{2,T}(1, 2 + 0.1*ϵ_Point) ≈ Point{2,T}(1,2)
            @test Point{2,T}(1, 2 + 10*ϵ_Point) ≉ Point{2,T}(1,2)
            
            P₁ = Point{2,T}(1, 2)
            P₂ = Point{2,T}(2, 4)
    
            # distance
            @test distance(P₁, P₂) ≈ sqrt(5)
    
            # distance²
            @test distance²(P₁, P₂) ≈ 5
    
            # midpoint
            mp = midpoint(P₁, P₂)
            @test mp ≈ [3//2, 3]

            # isCCW
            @test  isCCW(Point{2,T}(0,0), Point{2,T}(1,0), Point{2,T}(1,  1))
            @test !isCCW(Point{2,T}(0,0), Point{2,T}(1,0), Point{2,T}(1, -1))
        end
    end
    
    @testset "Point{3}" begin
        for T ∈ Floats 
            P₁ = Point{3,T}(1, 1, 0)
            P₂ = Point{3,T}(1, 0, 1)
            𝘃₂ = Vec{3,T}(1, 0, 1)

            # coordinates
            coords = coordinates(P₁)
            @test coords == [1, 1, 0]
    
            # subtraction
            P = P₁ - P₂
            @test P == [0, 1, -1]

            # vector addition
            v = P₁ + 𝘃₂
            @test v == [2, 1, 1]

            # vector subtraction 
            v = P₁ - 𝘃₂
            @test v == [0, 1, -1]
    
            # ≈
            @test Point{3,T}(1, 1, 2 + 0.1*ϵ_Point) ≈ Point{3,T}(1,1,2)
            @test Point{3,T}(1, 1, 2 + 10*ϵ_Point) ≉ Point{3,T}(1,1,2)
    
            P₁ = Point{3,T}(1, 2, 1)
            P₂ = Point{3,T}(2, 4, 0)
    
            # distance
            @test distance(P₁, P₂) ≈ sqrt(6)
    
            # distance²
            @test distance²(P₁, P₂) ≈ 6
    
            # midpoint
            mp = midpoint(P₁, P₂)
            @test mp ≈ [3//2, 3, 1//2]
        end
    end
end
