@testset "Interpolation" begin
    @testset "LineSegment" begin
        for T in Floats
            P₁ = Point{3,T}(1, 1, 1)
            P₂ = Point{3,T}(3, 3, 3)
            l = LineSegment(P₁, P₂)
            @test l(0) ≈ P₁
            @test l(1) ≈ P₂
            @test l(1//2) ≈ Point{3,T}(2, 2, 2)
        end
    end

    @testset "QuadraticSegment" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(2, 0, 0)
            P₃ = Point{3,T}(1, 1, 0)
            q = QuadraticSegment(P₁, P₂, P₃)
            for r = LinRange{T}(0, 1, 5)
                @test q(r) ≈ Point{3,T}(2r, -(2r)^2 + 4r, 0)
            end
        end
    end

    @testset "Triangle" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(0, 1, 0)
            P₃ = Point{3,T}(0, 0, 1)
            tri = Triangle(P₁, P₂, P₃)
            @test tri(0, 0) ≈ P₁
            @test tri(1, 0) ≈ P₂
            @test tri(0, 1) ≈ P₃
            @test tri(1//2, 1//2) ≈ Point{3,T}(0, 1//2, 1//2)
        end
    end

    @testset "Quadrilateral" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(0, 1, 0)
            P₃ = Point{3,T}(0, 1, 1)
            P₄ = Point{3,T}(0, 0, 1)
            quad = Quadrilateral(P₁, P₂, P₃, P₄)
            @test quad(0, 0) ≈ P₁
            @test quad(1, 0) ≈ P₂
            @test quad(1, 1) ≈ P₃
            @test quad(0, 1) ≈ P₄
            @test quad(1//2, 1//2) ≈ Point{3,T}(0, 1//2, 1//2)
        end
    end

    @testset "QuadraticTriangle" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(0, 1, 0)
            P₃ = Point{3,T}(0, 1, 1)
            P₄ = Point{3,T}(0, 1//2, 0)
            P₅ = Point{3,T}(0, 1, 1//2)
            P₆ = Point{3,T}(0, 1//2, 1//2)
            tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
            @test tri6(0, 0) ≈ P₁
            @test tri6(1, 0) ≈ P₂
            @test tri6(0, 1) ≈ P₃
            @test tri6(1//2, 0) ≈ P₄
            @test tri6(1//2, 1//2) ≈ P₅
            @test tri6(0, 1//2) ≈ P₆
        end
    end

    @testset "QuadraticQuadrilateral" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(0, 1, 0)
            P₃ = Point{3,T}(0, 1, 1)
            P₄ = Point{3,T}(0, 0, 1)
            P₅ = Point{3,T}(0, 1//2,    0)
            P₆ = Point{3,T}(0,    1, 1//2)
            P₇ = Point{3,T}(0, 1//2,    1)
            P₈ = Point{3,T}(0,    0, 1//2)
            quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            @test quad8(0, 0) ≈ P₁
            @test quad8(1, 0) ≈ P₂
            @test quad8(1, 1) ≈ P₃
            @test quad8(0, 1) ≈ P₄
            @test quad8(1//2,    0) ≈ P₅
            @test quad8(   1, 1//2) ≈ P₆
            @test quad8(1//2,    1) ≈ P₇
            @test quad8(   0, 1//2) ≈ P₈
            @test quad8(1//2, 1//2) ≈ Point{3,T}(0, 1//2, 1//2)
        end
    end

    @testset "Tetrahedron" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)           
            P₂ = Point{3,T}(1, 0, 0)
            P₃ = Point{3,T}(0, 1, 0)
            P₄ = Point{3,T}(0, 0, 1)
            tet = Tetrahedron(P₁, P₂, P₃, P₄)
            @test tet(0, 0, 0) ≈ P₁
            @test tet(1, 0, 0) ≈ P₂
            @test tet(0, 1, 0) ≈ P₃
            @test tet(0, 0, 1) ≈ P₄
        end
    end

    @testset "Hexahedron" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)
            P₃ = Point{3,T}(0, 1, 0)
            P₄ = Point{3,T}(1, 1, 0)
            P₅ = Point{3,T}(0, 0, 1)
            P₆ = Point{3,T}(1, 0, 1)
            P₇ = Point{3,T}(0, 1, 1)
            P₈ = Point{3,T}(1, 1, 1)
            hex = Hexahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            @test hex(0, 0, 0) ≈ P₁
            @test hex(1, 0, 0) ≈ P₂
            @test hex(1, 1, 0) ≈ P₃
            @test hex(0, 1, 0) ≈ P₄
            @test hex(0, 0, 1) ≈ P₅
            @test hex(1, 0, 1) ≈ P₆
            @test hex(1, 1, 1) ≈ P₇
            @test hex(0, 1, 1) ≈ P₈
        end
    end

    @testset "QuadraticTetrahedron" begin
        for T in Floats
            P₁  = Point{3,T}(0, 0, 0)
            P₂  = Point{3,T}(1, 0, 0)
            P₃  = Point{3,T}(0, 1, 0)
            P₄  = Point{3,T}(0, 0, 1)
            P₅  = Point{3,T}(1//2,    0,    0)
            P₆  = Point{3,T}(1//2, 1//2,    0)
            P₇  = Point{3,T}(0,    1//2,    0)
            P₈  = Point{3,T}(0,       0, 1//2)
            P₉  = Point{3,T}(1//2,    0, 1//2)
            P₁₀ = Point{3,T}(0,    1//2, 1//2)
            tet10 = QuadraticTetrahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, P₁₀)
            @test tet10(0, 0, 0) ≈ P₁ 
            @test tet10(1, 0, 0) ≈ P₂ 
            @test tet10(0, 1, 0) ≈ P₃ 
            @test tet10(0, 0, 1) ≈ P₄ 
            @test tet10(1//2,    0,    0) ≈ P₅
            @test tet10(1//2, 1//2,    0) ≈ P₆ 
            @test tet10(0,    1//2,    0) ≈ P₇ 
            @test tet10(0,       0, 1//2) ≈ P₈ 
            @test tet10(1//2,    0, 1//2) ≈ P₉ 
            @test tet10(0,    1//2, 1//2) ≈ P₁₀ 
        end
    end
    
    @testset "QuadraticHexahedon" begin
        for T in Floats
            P₁  = Point{3,T}(0, 0, 0)
            P₂  = Point{3,T}(1, 0, 0)
            P₃  = Point{3,T}(1, 1, 0)
            P₄  = Point{3,T}(0, 1, 0)
            P₅  = Point{3,T}(0, 0, 1)
            P₆  = Point{3,T}(1, 0, 1)
            P₇  = Point{3,T}(1, 1, 1)
            P₈  = Point{3,T}(0, 1, 1)
            P₉  = Point{3,T}(1//2,    0,    0)
            P₁₀ = Point{3,T}(   1, 1//2,    0)
            P₁₁ = Point{3,T}(1//2,    1,    0)
            P₁₂ = Point{3,T}(   0, 1//2,    0)
            P₁₃ = Point{3,T}(1//2,    0,    1)
            P₁₄ = Point{3,T}(   1, 1//2,    1)
            P₁₅ = Point{3,T}(1//2,    1,    1)
            P₁₆ = Point{3,T}(   0, 1//2,    1)
            P₁₇ = Point{3,T}(0, 0, 1//2)
            P₁₈ = Point{3,T}(1, 0, 1//2)
            P₁₉ = Point{3,T}(1, 1, 1//2)
            P₂₀ = Point{3,T}(0, 0, 1//2)
            hex20 = QuadraticHexahedron(P₁,  P₂,  P₃,  P₄,  P₅,  P₆,  P₇,
                                        P₈,  P₉,  P₁₀, P₁₁, P₁₂, P₁₃, P₁₄, 
                                        P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
            @test hex20(0, 0, 0) ≈ P₁
            @test hex20(1, 0, 0) ≈ P₂
            @test hex20(1, 1, 0) ≈ P₃
            @test hex20(0, 1, 0) ≈ P₄
            @test hex20(0, 0, 1) ≈ P₅
            @test hex20(1, 0, 1) ≈ P₆
            @test hex20(1, 1, 1) ≈ P₇
            @test hex20(0, 1, 1) ≈ P₈
            @test hex20(1//2,    0,    0)≈ P₉
            @test hex20(   1, 1//2,    0)≈ P₁₀    
            @test hex20(1//2,    1,    0)≈ P₁₁
            @test hex20(   0, 1//2,    0)≈ P₁₂
            @test hex20(1//2,    0,    1)≈ P₁₃
            @test hex20(   1, 1//2,    1)≈ P₁₄
            @test hex20(1//2,    1,    1)≈ P₁₅
            @test hex20(   0, 1//2,    1)≈ P₁₆
            @test hex20(0, 0, 1//2)≈ P₁₇
            @test hex20(1, 0, 1//2)≈ P₁₈
            @test hex20(1, 1, 1//2)≈ P₁₉
            @test hex20(0, 0, 1//2)≈ P₂₀
        end
    end
end
