@testset "Polytope" begin
    ###########################################################################
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
                P₁ = Point{2,T}(1, 0, 1)
                P₂ = Point{2,T}(2, 0, -1) 
                l = LineSegment(P₁, P₂) 
                @test l.vertices[1] == P₁
                @test l.vertices[2] == P₂
            end
        end 

        @testset "QuadraticSegment{2}" begin
            for T ∈ Floats
                # Constructor
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(2, 0)
                P₃ = Point{2,T}(1, 1)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test q.vertices == Vec(P₁, P₂, P₃)

                # isstraight
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(2, 0)
                P₃ = Point{2,T}(1, 0)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test isstraight(q)
                P₂ = Point{2,T}(2, 0.001)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test !isstraight(q)
            end
        end

        @testset "QuadraticSegment{3}" begin
            for T ∈ Floats
                # Constructor
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(0, 2, 0)
                P₃ = Point{3,T}(0, 1, 1)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test q.vertices == Vec(P₁, P₂, P₃)

                # isstraight
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(0, 2, 0)
                P₃ = Point{3,T}(0, 1, 0)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test isstraight(q)
                P₂ = Point{3,T}(0, 2, 0.001)
                q = QuadraticSegment(P₁, P₂, P₃)
                @test !isstraight(q)
            end
        end
    end
    ###########################################################################
    @testset "Polytope{2}" begin
        @testset "Triangle{2}" begin
            for T in Floats
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(1, 0)
                P₃ = Point{2,T}(0, 1)
                tri = Triangle(P₁, P₂, P₃)
                @test tri.vertices == Vec(P₁, P₂, P₃)
            end
        end

        @testset "Triangle{3}" begin
            for T in Floats
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(0, 1, 0)
                P₃ = Point{3,T}(0, 0, 1)
                tri = Triangle(P₁, P₂, P₃)
                @test tri.vertices == Vec(P₁, P₂, P₃)
            end
        end

        @testset "Quadrilateral{2}" begin
            for T in Floats
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(1, 0)
                P₃ = Point{2,T}(1, 1)
                P₄ = Point{2,T}(0, 1)
                quad = Quadrilateral(P₁, P₂, P₃, P₄)
                @test quad.vertices == Vec(P₁, P₂, P₃, P₄)
            end
        end

        @testset "Quadrilateral{3}" begin
            for T in Floats
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(0, 1, 0)
                P₃ = Point{3,T}(0, 1, 1)
                P₄ = Point{3,T}(0, 0, 1)
                quad = Quadrilateral(P₁, P₂, P₃, P₄)
                @test quad.vertices == Vec(P₁, P₂, P₃, P₄)
            end
        end

        @testset "QuadraticTriangle{2}" begin
            for T in Floats
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(1, 0)
                P₃ = Point{2,T}(1, 1)
                P₄ = Point{2,T}(1//2, 0)
                P₅ = Point{2,T}(1, 1//2)
                P₆ = Point{2,T}(1//2, 1//2)
                tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
                @test tri6.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆)
            end
        end

        @testset "QuadraticQuadrilateral{2}" begin
            for T in Floats
                P₁ = Point{2,T}(0, 0)
                P₂ = Point{2,T}(1, 0)
                P₃ = Point{2,T}(1, 1)
                P₄ = Point{2,T}(0, 1)
                P₅ = Point{2,T}(1//2,    0)
                P₆ = Point{2,T}(   1, 1//2)
                P₇ = Point{2,T}(1//2,    1)
                P₈ = Point{2,T}(   0, 1//2)
                quad8 = QuadraticQuadrilateral(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
                @test quad8.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            end
        end

        @testset "QuadraticTriangle{3}" begin
            for T in Floats
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(0, 1, 0)
                P₃ = Point{3,T}(0, 1, 1)
                P₄ = Point{3,T}(0, 1//2, 0)
                P₅ = Point{3,T}(0, 1, 1//2)
                P₆ = Point{3,T}(0, 1//2, 1//2)
                tri6 = QuadraticTriangle(P₁, P₂, P₃, P₄, P₅, P₆)
                @test tri6.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆)
            end
        end

        @testset "QuadraticQuadrilateral{3}" begin
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
                @test quad8.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            end
        end
    end
    ###########################################################################
    @testset "Polytope{3}" begin
        @testset "Tetrahedron" begin
            for T in Floats
                P₁ = Point{3,T}(0, 0, 0)
                P₂ = Point{3,T}(1, 0, 0)
                P₃ = Point{3,T}(0, 1, 0)
                P₄ = Point{3,T}(0, 0, 1)
                tet = Tetrahedron(P₁, P₂, P₃, P₄)
                @test tet.vertices == Vec(P₁, P₂, P₃, P₄)
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
                @test hex.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
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
                @test tet10.vertices == Vec(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, P₁₀)
            end
        end

        @testset "QuadraticHexahedon" begin
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
                P₁₁ = Point{3,T}(0, 0, 0)
                P₁₂ = Point{3,T}(1, 0, 0)
                P₁₃ = Point{3,T}(0, 1, 0)
                P₁₄ = Point{3,T}(0, 0, 1)
                P₁₅ = Point{3,T}(1//2,    0,    0)
                P₁₆ = Point{3,T}(1//2, 1//2,    0)
                P₁₇ = Point{3,T}(0,    1//2,    0)
                P₁₈ = Point{3,T}(0,       0, 1//2)
                P₁₉ = Point{3,T}(1//2,    0, 1//2)
                P₂₀ = Point{3,T}(0,    1//2, 1//2)
                hex20 = QuadraticHexahedron(P₁,  P₂,  P₃,  P₄,  P₅,  P₆,  P₇,
                                            P₈,  P₉,  P₁₀, P₁₁, P₁₂, P₁₃, P₁₄,
                                            P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
                @test hex20.vertices == Vec(P₁,  P₂,  P₃,  P₄,  P₅,  P₆,  P₇,
                                            P₈,  P₉,  P₁₀, P₁₁, P₁₂, P₁₃, P₁₄,
                                            P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
            end
        end
    end
end
