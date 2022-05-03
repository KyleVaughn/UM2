@testset "Faces" begin
    @testset "Tetrahedron" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)
            P₃ = Point{3,T}(0, 1, 0)
            P₄ = Point{3,T}(0, 0, 1)
            tet = Tetrahedron(P₁, P₂, P₃, P₄)
            tet_faces = faces(tet)
            @test length(tet_faces) == 4
            @test tet_faces[1] == Triangle(P₁, P₂, P₃)  
            @test tet_faces[2] == Triangle(P₁, P₂, P₄) 
            @test tet_faces[3] == Triangle(P₂, P₃, P₄) 
            @test tet_faces[4] == Triangle(P₃, P₁, P₄) 
        end
    end

    @testset "Hexahedron" begin
        for T in Floats
            P₁ = Point{3,T}(0, 0, 0)
            P₂ = Point{3,T}(1, 0, 0)    
            P₃ = Point{3,T}(1, 1, 0)    
            P₄ = Point{3,T}(0, 1, 0)    
            P₅ = Point{3,T}(0, 0, 1)
            P₆ = Point{3,T}(1, 0, 1)    
            P₇ = Point{3,T}(1, 1, 1)    
            P₈ = Point{3,T}(0, 1, 1)    
            hex = Hexahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈) 
            v = (P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈)
            hex_faces = faces(hex)
            @test length(hex_faces) == 6
            @test hex_faces[1] == Quadrilateral(v[1], v[2], v[3], v[4])
            @test hex_faces[2] == Quadrilateral(v[5], v[6], v[7], v[8])
            @test hex_faces[3] == Quadrilateral(v[1], v[2], v[6], v[5])
            @test hex_faces[4] == Quadrilateral(v[2], v[3], v[7], v[6])
            @test hex_faces[5] == Quadrilateral(v[3], v[4], v[8], v[7])
            @test hex_faces[6] == Quadrilateral(v[4], v[1], v[5], v[8])
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
            tet = QuadraticTetrahedron(P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, P₁₀)
            v = (P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, P₁₀)
            tet_faces = faces(tet)
            @test length(tet_faces) == 4
            @test tet_faces[1] == QuadraticTriangle(v[1], v[2], v[3], v[5],  v[6],  v[7])
            @test tet_faces[2] == QuadraticTriangle(v[1], v[2], v[4], v[5],  v[9],  v[8])
            @test tet_faces[3] == QuadraticTriangle(v[2], v[3], v[4], v[6], v[10],  v[9])
            @test tet_faces[4] == QuadraticTriangle(v[3], v[1], v[4], v[7],  v[8], v[10]) 
        end
    end

    @testset "QuadraticHexahedron" begin
        for T in Floats
            P₁  = Point{3,T}(   0,    0,    0)
            P₂  = Point{3,T}(   1,    0,    0)
            P₃  = Point{3,T}(   1,    1,    0)
            P₄  = Point{3,T}(   0,    1,    0)
            P₅  = Point{3,T}(   0,    0,    1)
            P₆  = Point{3,T}(   1,    0,    1)
            P₇  = Point{3,T}(   1,    1,    1)
            P₈  = Point{3,T}(   0,    1,    1)
            P₉  = Point{3,T}(1//2,    0,    0)
            P₁₀ = Point{3,T}(   1, 1//2,    0)
            P₁₁ = Point{3,T}(1//2,    1,    0)
            P₁₂ = Point{3,T}(   0, 1//2,    0)
            P₁₃ = Point{3,T}(1//2,    0,    1)
            P₁₄ = Point{3,T}(   1, 1//2,    1)
            P₁₅ = Point{3,T}(1//2,    1,    1)
            P₁₆ = Point{3,T}(   0, 1//2,    1)
            P₁₇ = Point{3,T}(   0,    0, 1//2)
            P₁₈ = Point{3,T}(   1,    0, 1//2)
            P₁₉ = Point{3,T}(   1,    1, 1//2)
            P₂₀ = Point{3,T}(   0,    1, 1//2)
            hex = QuadraticHexahedron(P₁,  P₂,  P₃,  P₄,  P₅,  P₆,  P₇,
                                      P₈,  P₉,  P₁₀, P₁₁, P₁₂, P₁₃, P₁₄,
                                      P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
            v = (P₁,  P₂,  P₃,  P₄,  P₅,  P₆,  P₇,
                 P₈,  P₉,  P₁₀, P₁₁, P₁₂, P₁₃, P₁₄,
                 P₁₅, P₁₆, P₁₇, P₁₈, P₁₉, P₂₀)
            hex_faces = faces(hex)
            @test length(hex_faces) == 6
            @test hex_faces[1] == QuadraticQuadrilateral(v[1], v[2], v[3], v[4], v[ 9], v[10], v[11], v[12])
            @test hex_faces[2] == QuadraticQuadrilateral(v[5], v[6], v[7], v[8], v[13], v[14], v[15], v[16])
            @test hex_faces[3] == QuadraticQuadrilateral(v[1], v[2], v[6], v[5], v[ 9], v[18], v[13], v[17])
            @test hex_faces[4] == QuadraticQuadrilateral(v[2], v[3], v[7], v[6], v[10], v[19], v[14], v[18])
            @test hex_faces[5] == QuadraticQuadrilateral(v[3], v[4], v[8], v[7], v[11], v[20], v[15], v[19])
            @test hex_faces[6] == QuadraticQuadrilateral(v[4], v[1], v[5], v[8], v[12], v[17], v[16], v[20])
        end
    end
end
