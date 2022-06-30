@testset "Jacobian" begin
    @testset "LineSegment" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        l = LineSegment(P₁, P₂)
        @test jacobian(l, 1) == [1, 0, 0]
        @test jacobian(l, 0) == [1, 0, 0]
    end end

    @testset "QuadraticSegment" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(1 // 2, 0, 0)
        q = QuadraticSegment(P₁, P₂, P₃)
        @test jacobian(q, 0) == [1, 0, 0]
        @test jacobian(q, 1 // 2) == [1, 0, 0]
        @test jacobian(q, 1) == [1, 0, 0]
    end end

    @testset "Triangle" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(0, 1, 0)
        tri = Triangle(P₁, P₂, P₃)
        @test jacobian(tri, 0, 0) == [1 0; 0 1; 0 0]
        @test jacobian(tri, 1, 0) == [1 0; 0 1; 0 0]
        @test jacobian(tri, 0, 1) == [1 0; 0 1; 0 0]
    end end

    @testset "Quadrilateral" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(1, 1, 0)
        P₄ = Point{3, T}(0, 1, 0)
        quad = Quadrilateral(P₁, P₂, P₃, P₄)
        @test jacobian(quad, 0, 0) == [1 0; 0 1; 0 0]
        @test jacobian(quad, 1, 0) == [1 0; 0 1; 0 0]
        @test jacobian(quad, 1, 1) == [1 0; 0 1; 0 0]
        @test jacobian(quad, 0, 1) == [1 0; 0 1; 0 0]
    end end

    @testset "Quadratic Triangle" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(0, 1, 0)
        P₄ = Point{3, T}(1 // 2, 0, 0)
        P₅ = Point{3, T}(1 // 2, 1 // 2, 0)
        P₆ = Point{3, T}(0, 1 // 2, 0)
        tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
        @test jacobian(tri6, 0, 0) == [1 0; 0 1; 0 0]
        @test jacobian(tri6, 1, 0) == [1 0; 0 1; 0 0]
        @test jacobian(tri6, 0, 1) == [1 0; 0 1; 0 0]
        @test jacobian(tri6, 1 // 2, 0) == [1 0; 0 1; 0 0]
        @test jacobian(tri6, 0, 1 // 2) == [1 0; 0 1; 0 0]
        @test jacobian(tri6, 1 // 2, 1 // 2) == [1 0; 0 1; 0 0]
    end end

    @testset "Quadratic Quadrilateral" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(1, 1, 0)
        P₄ = Point{3, T}(0, 1, 0)
        P₅ = Point{3, T}(1 // 2, 0, 0)
        P₆ = Point{3, T}(1, 1 // 2, 0)
        P₇ = Point{3, T}(1 // 2, 1, 0)
        P₈ = Point{3, T}(0, 1 // 2, 0)
        quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
        @test jacobian(quad8, 0, 0) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 1, 0) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 1, 1) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 0, 1) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 1 // 2, 0) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 1, 1 // 2) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 1 // 2, 1) == [1 0; 0 1; 0 0]
        @test jacobian(quad8, 0, 1 // 2) == [1 0; 0 1; 0 0]
    end end

    @testset "Tetrahedron" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(0, 1, 0)
        P₄ = Point{3, T}(0, 0, 1)
        tet = Tetrahedron(P₁, P₂, P₃, P₄)
        @test jacobian(tet, 0, 0, 0) == I
        @test jacobian(tet, 1, 0, 0) == I
        @test jacobian(tet, 0, 1, 0) == I
        @test jacobian(tet, 0, 0, 1) == I
    end end

    @testset "Hexahedron" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(1, 1, 0)
        P₄ = Point{3, T}(0, 1, 0)
        P₅ = Point{3, T}(0, 0, 1)
        P₆ = Point{3, T}(1, 0, 1)
        P₇ = Point{3, T}(1, 1, 1)
        P₈ = Point{3, T}(0, 1, 1)
        hex = Hexahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
        @test jacobian(hex, 0, 0, 0) == I
        @test jacobian(hex, 1, 0, 0) == I
        @test jacobian(hex, 1, 1, 0) == I
        @test jacobian(hex, 0, 1, 0) == I
        @test jacobian(hex, 0, 0, 1) == I
        @test jacobian(hex, 1, 0, 1) == I
        @test jacobian(hex, 1, 1, 1) == I
        @test jacobian(hex, 0, 1, 1) == I
    end end

    @testset "QuadraticTetrahedron" begin for T in Floats
        P₁  = Point{3, T}(0, 0, 0)
        P₂  = Point{3, T}(1, 0, 0)
        P₃  = Point{3, T}(0, 1, 0)
        P₄  = Point{3, T}(0, 0, 1)
        P₅  = Point{3, T}(1 // 2, 0, 0)
        P₆  = Point{3, T}(1 // 2, 1 // 2, 0)
        P₇  = Point{3, T}(0, 1 // 2, 0)
        P₈  = Point{3, T}(0, 0, 1 // 2)
        P₉  = Point{3, T}(1 // 2, 0, 1 // 2)
        P₁₀ = Point{3, T}(0, 1 // 2, 1 // 2)
        tet = QuadraticTetrahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, P₁₀)
        @test jacobian(tet, 0, 0, 0) == I
        @test jacobian(tet, 1, 0, 0) == I
        @test jacobian(tet, 0, 1, 0) == I
        @test jacobian(tet, 0, 0, 1) == I
        @test jacobian(tet, 1 // 2, 0, 0) == I
        @test jacobian(tet, 1 // 2, 1 // 2, 0) == I
        @test jacobian(tet, 0, 1 // 2, 0) == I
        @test jacobian(tet, 0, 0, 1 // 2) == I
        @test jacobian(tet, 1 // 2, 0, 1 // 2) == I
        @test jacobian(tet, 0, 1 // 2, 1 // 2) == I
        if T != BigFloat
            @test @ballocated(jacobian($tet, 0, 0, 0), samples = 1, evals = 2) == 0
        end
    end end

    @testset "QuadraticHexahedon" begin for T in Floats
        P₁ = Point{3, T}(0, 0, 0)
        P₂ = Point{3, T}(1, 0, 0)
        P₃ = Point{3, T}(1, 1, 0)
        P₄ = Point{3, T}(0, 1, 0)
        P₅ = Point{3, T}(0, 0, 1)
        P₆ = Point{3, T}(1, 0, 1)
        P₇ = Point{3, T}(1, 1, 1)
        P₈ = Point{3, T}(0, 1, 1)
        P₉ = Point{3, T}(1 // 2, 0, 0)
        P₁₀ = Point{3, T}(1, 1 // 2, 0)
        P₁₁ = Point{3, T}(1 // 2, 1, 0)
        P₁₂ = Point{3, T}(0, 1 // 2, 0)
        P₁₃ = Point{3, T}(1 // 2, 0, 1)
        P₁₄ = Point{3, T}(1, 1 // 2, 1)
        P₁₅ = Point{3, T}(1 // 2, 1, 1)
        P₁₆ = Point{3, T}(0, 1 // 2, 1)
        P₁₇ = Point{3, T}(0, 0, 1 // 2)
        P₁₈ = Point{3, T}(1, 0, 1 // 2)
        P₁₉ = Point{3, T}(1, 1, 1 // 2)
        P₂₀ = Point{3, T}(0, 0, 1 // 2)
        hex20 = QuadraticHexahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, 
                                    P₈, P₉, P₁₀, P₁₁, P₁₂, P₁₃, P₁₄,
                                    P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
        # jacobian for hex20 not yet implemented due to algebraic chore of deriving
        # the jacobian.
#        @test jacobian(hex20, 0, 0, 0) == I 
#        @test jacobian(hex20, 1, 0, 0) == I 
#        @test jacobian(hex20, 1, 1, 0) == I 
#        @test jacobian(hex20, 0, 1, 0) == I 
#        @test jacobian(hex20, 0, 0, 1) == I 
#        @test jacobian(hex20, 1, 0, 1) == I 
#        @test jacobian(hex20, 1, 1, 1) == I 
#        @test jacobian(hex20, 0, 1, 1) == I 
#        @test jacobian(hex20, 1 // 2, 0, 0) == I 
#        @test jacobian(hex20, 1, 1 // 2, 0) == I 
#        @test jacobian(hex20, 1 // 2, 1, 0) == I 
#        @test jacobian(hex20, 0, 1 // 2, 0) == I 
#        @test jacobian(hex20, 1 // 2, 0, 1) == I 
#        @test jacobian(hex20, 1, 1 // 2, 1) == I 
#        @test jacobian(hex20, 1 // 2, 1, 1) == I 
#        @test jacobian(hex20, 0, 1 // 2, 1) == I 
#        @test jacobian(hex20, 0, 0, 1 // 2) == I 
#        @test jacobian(hex20, 1, 0, 1 // 2) == I 
#        @test jacobian(hex20, 1, 1, 1 // 2) == I 
#        @test jacobian(hex20, 0, 0, 1 // 2) == I 
    end end 
end
