@testset "Jacobian" begin
    @testset "LineSegment" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(3, 4, 0)    
            l = LineSegment(P₁, P₂)
            @test jacobian(l, 1) == [3,4,0]
            @test jacobian(l, 0) == [3,4,0]
            weights, points = gauss_quadrature(Val(:legendre), RefLine(),
                                               Val(1), T)
            l_measure = 5
            
            for i = 1:length(weights)
        end
    end

    @testset "QuadraticSegment" begin
        for T ∈ Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(2, 0, 0)    
            P₃ = Point{3,T}(1, 1, 0)    
            q = QuadraticSegment(P₁, P₂, P₃)
            @test jacobian(q, 0)    == [3,4,0]
            @test jacobian(q, 1//2) == [3,4,0]
            @test jacobian(q, 1)    == [3,4,0]
        end
    end
end
