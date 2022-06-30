@testset "Centroid" begin
    @testset "Triangle" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(0, 1)
        tri = Triangle(P₁, P₂, P₃)
        @test centroid(tri) ≈ Point{2, T}(1//3, 1//3)
        if T != BigFloat
            @test @ballocated(centroid($tri), samples = 1, evals = 2) == 0
        end
    end end

    @testset "Quadrilateral" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(1, 1)
        P₄ = Point{2, T}(0, 1)
        quad = Quadrilateral(P₁, P₂, P₃, P₄)
        @test centroid(quad) ≈ Point{2, T}(1//2, 1//2)
        if T != BigFloat
            @test @ballocated(centroid($quad), samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticTriangle" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(0, 1)
        P₄ = Point{2, T}(1 // 2, 0)
        P₅ = Point{2, T}(1//2, 1 // 2)
        P₆ = Point{2, T}(0, 1 // 2)
        tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
        @test centroid(tri6) ≈ Point{2, T}(1//3, 1//3)
        if T != BigFloat
            @test @ballocated(centroid($tri6), samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticQuadrilateral" begin for T in Floats
        P₁ = Point{2, T}(0, 0)
        P₂ = Point{2, T}(1, 0)
        P₃ = Point{2, T}(1, 1)
        P₄ = Point{2, T}(0, 1)
        P₅ = Point{2, T}(1 // 2, 0)
        P₆ = Point{2, T}(1, 1 // 2)
        P₇ = Point{2, T}(1 // 2, 1)
        P₈ = Point{2, T}(0, 1 // 2)
        quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
        @test centroid(quad8) ≈ Point{2, T}(1//2, 1//2)
        if T != BigFloat
            @test @ballocated(centroid($quad8), samples = 1, evals = 2) == 0
        end
    end end
end
