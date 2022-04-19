@testset "Polytope" begin
    @testset "Polytope{1}" begin
        @testset "LineSegment{2}" begin
            for T ∈ Floats
                P₁ = Point{2,T}(1, 0)
                P₂ = Point{2,T}(2, 0)
                l = LineSegment(P₁, P₂) 
                @test l.vertices[1] == P₁
                @test l.vertices[2] == P₂
            end
        end 
        
        @testset "LineSegment{3}" begin
            for T ∈ Floats
                p₁ = Point{2,T}(1, 0, 1)
                p₂ = Point{2,T}(2, 0, -1) 
                l = LineSegment(P₁, P₂) 
                @test l.vertices[1] == P₁
                @test l.vertices[2] == P₂
            end
        end 
    end
end

