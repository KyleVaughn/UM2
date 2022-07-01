@testset "In" begin
    @testset "isleft" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(2, 0)
        P_in = Point{2, T}(0, 2)
        P_out = Point{2, T}(0, -2)

        l = LineSegment(P₁, P₂)
        @test isleft(Point{2, T}(0, 1), l)  
        @test !isleft(Point{2, T}(0, -1), l)  

        q₁ = QuadraticSegment(P₁, P₂, Point{2, T}(1, 0))
        q₂ = QuadraticSegment(P₁, P₂, Point{2, T}(1, 1))
        @test isleft(P_in, q₁)  
        @test !isleft(P_out, q₁)  
        @test isleft(P_in, q₂)  
        @test !isleft(P_out, q₂)  

        if T != BigFloat
            @test @ballocated(isleft($P_in, $l) , samples = 1, evals = 2) == 0
            @test @ballocated(isleft($P_in, $q₁), samples = 1, evals = 2) == 0 
            @test @ballocated(isleft($P_in, $q₂), samples = 1, evals = 2) == 0
        end
    end end

    @testset "Triangle" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(0, 1)
        P_in = Point{2, T}(1//4, 1//4)
        P_out = Point{2, T}(1//4, -1//4)
        tri = Triangle(P₁, P₂, P₃)
        @test P_in ∈ tri 
        @test P_out ∉ tri 
        if T != BigFloat
            @test @ballocated($P_in ∈ $tri, samples = 1, evals = 2) == 0
        end
    end end

    @testset "Quadrilateral" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(1, 1)
        P₄ = Point{2, T}(0, 1)
        quad = Quadrilateral(P₁, P₂, P₃, P₄)
        P_in = Point{2, T}(1//4, 1//4)
        P_out = Point{2, T}(1//4, -1//4)
        @test P_in ∈ quad 
        @test P_out ∉ quad 
        if T != BigFloat
            @test @ballocated($P_in ∈ $quad, samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticTriangle" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(0, 1)
        P₄ = Point{2, T}(1 // 3, -1//5)
        P₅ = Point{2, T}(1//2, 1 // 2)
        P₆ = Point{2, T}(0, 1 // 2)
        P_in = Point{2, T}(1//4, 1//4)
        P_out = Point{2, T}(1//4, -1//4)

        tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
        @test P_in ∈ tri6 
        @test P_out ∉ tri6 
        if T != BigFloat
            @test @ballocated($P_in ∈ $tri6, samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticQuadrilateral" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(1, 1)
        P₄ = Point{2, T}(0, 1)
        P₅ = Point{2, T}(1 // 3, -1//5)
        P₆ = Point{2, T}(1, 1 // 2)
        P₇ = Point{2, T}(2 // 3, 6//5)
        P₈ = Point{2, T}(0, 1 // 2)
        quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
        P_in = Point{2, T}(1//4, 1//4)
        P_out = Point{2, T}(1//4, -1//4)
        @test P_in ∈ quad8 
        @test P_out ∉ quad8 
        if T != BigFloat
            @test @ballocated($P_in ∈ $quad8, samples = 1, evals = 2) == 0
        end
    end end
end
