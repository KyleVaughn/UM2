@testset "Edges" begin
    @testset "Triangle" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(0, 1, 0)
        P₃ = Point{3, T}(0, 0, 1)
        tri = Triangle(P₁, P₂, P₃)
        tri_edges = edges(tri)
        @test length(tri_edges) == 3
        @test tri_edges[1] == LineSegment(P₁, P₂)
        @test tri_edges[2] == LineSegment(P₂, P₃)
        @test tri_edges[3] == LineSegment(P₃, P₁)
    end end

    @testset "QuadraticTriangle" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(0, 1, 0)
        P₃ = Point{3, T}(0, 1, 1)
        P₄ = Point{3, T}(0, 1 // 2, 0)
        P₅ = Point{3, T}(0, 1, 1 // 2)
        P₆ = Point{3, T}(0, 1 // 2, 1 // 2)
        tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
        tri6_edges = edges(tri6)
        @test length(tri6_edges) == 3
        @test tri6_edges[1] == QuadraticSegment(P₁, P₂, P₄)
        @test tri6_edges[2] == QuadraticSegment(P₂, P₃, P₅)
        @test tri6_edges[3] == QuadraticSegment(P₃, P₁, P₆)
    end end
end
