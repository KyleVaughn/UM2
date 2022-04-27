@testset "Jacobian" begin
    @testset "LineSegment" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)    
            l = LineSegment(P₁, P₂)
            @test jacobian(l, 1) == [1,0,0]
            @test jacobian(l, 0) == [1,0,0]
        end
    end

    @testset "QuadraticSegment" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)    
            P₃ = Point{3,T}(1//2, 0, 0)    
            q = QuadraticSegment(P₁, P₂, P₃)
            @test jacobian(q, 0)    == [1,0,0]
            @test jacobian(q, 1//2) == [1,0,0]
            @test jacobian(q, 1)    == [1,0,0]
        end
    end

    @testset "Triangle" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)    
            P₃ = Point{3,T}(0, 1, 0)                
            tri = Triangle(P₁, P₂, P₃)
            @test jacobian(tri, 0, 0) == [1 0; 0 1; 0 0]
            @test jacobian(tri, 1, 0) == [1 0; 0 1; 0 0]
            @test jacobian(tri, 0, 1) == [1 0; 0 1; 0 0]
        end
    end

    @testset "Quadrilateral" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)    
            P₃ = Point{3,T}(1, 1, 0)                
            P₄ = Point{3,T}(0, 1, 0)                
            quad = Quadrilateral(P₁, P₂, P₃, P₄)
            @test jacobian(quad, 0, 0) == [1 0; 0 1; 0 0]
            @test jacobian(quad, 1, 0) == [1 0; 0 1; 0 0]
            @test jacobian(quad, 1, 1) == [1 0; 0 1; 0 0]
            @test jacobian(quad, 0, 1) == [1 0; 0 1; 0 0]
        end
    end

    @testset "Quadratic Triangle" begin
        for T ∈ Floats
            P₁ = Point{3,T}(   0,    0, 0)
            P₂ = Point{3,T}(   1,    0, 0)    
            P₃ = Point{3,T}(   0,    1, 0)                
            P₄ = Point{3,T}(1//2,    0, 0)                
            P₅ = Point{3,T}(1//2, 1//2, 0)                
            P₆ = Point{3,T}(   0, 1//2, 0)                
            tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
            @test jacobian(tri6,    0,    0) == [1 0; 0 1; 0 0]
            @test jacobian(tri6,    1,    0) == [1 0; 0 1; 0 0]
            @test jacobian(tri6,    0,    1) == [1 0; 0 1; 0 0]
            @test jacobian(tri6, 1//2,    0) == [1 0; 0 1; 0 0]
            @test jacobian(tri6,    0, 1//2) == [1 0; 0 1; 0 0]
            @test jacobian(tri6, 1//2, 1//2) == [1 0; 0 1; 0 0]
        end
    end

    @testset "Quadratic Quadrilateral" begin
        for T ∈ Floats
            P₁ = Point{3,T}(   0,    0, 0)
            P₂ = Point{3,T}(   1,    0, 0)    
            P₃ = Point{3,T}(   1,    1, 0)                
            P₄ = Point{3,T}(   0,    1, 0)                
            P₅ = Point{3,T}(1//2,    0, 0)                
            P₆ = Point{3,T}(   1, 1//2, 0)                
            P₇ = Point{3,T}(1//2,    1, 0)                
            P₈ = Point{3,T}(   0, 1//2, 0)                
            quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            @test jacobian(quad8,    0,    0) == [1 0; 0 1; 0 0]
            @test jacobian(quad8,    1,    0) == [1 0; 0 1; 0 0]
            @test jacobian(quad8,    1,    1) == [1 0; 0 1; 0 0]
            @test jacobian(quad8,    0,    1) == [1 0; 0 1; 0 0]
            @test jacobian(quad8, 1//2,    0) == [1 0; 0 1; 0 0]
            @test jacobian(quad8,    1, 1//2) == [1 0; 0 1; 0 0]
            @test jacobian(quad8, 1//2,    1) == [1 0; 0 1; 0 0]
            @test jacobian(quad8,    0, 1//2) == [1 0; 0 1; 0 0]
        end
    end
end
