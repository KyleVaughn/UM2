@testset "QuadraticPolyhedron" begin
    @testset "QuadraticTetrahedron" begin
        for T in Floats
            p₁  = Point3D{T}(0, 0, 0)
            p₂  = Point3D{T}(1, 0, 0)
            p₃  = Point3D{T}(0, 1, 0)
            p₄  = Point3D{T}(0, 0, 1)
            p₅  = Point3D{T}(1//2,    0,    0)
            p₆  = Point3D{T}(1//2, 1//2,    0)
            p₇  = Point3D{T}(0,    1//2,    0)
            p₈  = Point3D{T}(0,       0, 1//2)
            p₉  = Point3D{T}(1//2,    0, 1//2)
            p₁₀ = Point3D{T}(0,    1//2, 1//2)
            tet10 = QuadraticTetrahedron(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈, p₉, p₁₀)
            @test tet10.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈, p₉, p₁₀)
        end
    end
    
    @testset "QuadraticHexahedon" begin
        for T in Floats
            p₁  = Point3D{T}(0, 0, 0)
            p₂  = Point3D{T}(1, 0, 0)
            p₃  = Point3D{T}(0, 1, 0)
            p₄  = Point3D{T}(0, 0, 1)
            p₅  = Point3D{T}(1//2,    0,    0)
            p₆  = Point3D{T}(1//2, 1//2,    0)
            p₇  = Point3D{T}(0,    1//2,    0)
            p₈  = Point3D{T}(0,       0, 1//2)
            p₉  = Point3D{T}(1//2,    0, 1//2)
            p₁₀ = Point3D{T}(0,    1//2, 1//2)
            p₁₁ = Point3D{T}(0, 0, 0)
            p₁₂ = Point3D{T}(1, 0, 0)
            p₁₃ = Point3D{T}(0, 1, 0)
            p₁₄ = Point3D{T}(0, 0, 1)
            p₁₅ = Point3D{T}(1//2,    0,    0)
            p₁₆ = Point3D{T}(1//2, 1//2,    0)
            p₁₇ = Point3D{T}(0,    1//2,    0)
            p₁₈ = Point3D{T}(0,       0, 1//2)
            p₁₉ = Point3D{T}(1//2,    0, 1//2)
            p₂₀ = Point3D{T}(0,    1//2, 1//2)
            hex20 = QuadraticHexahedron(p₁,  p₂,  p₃,  p₄,  p₅,  p₆,  p₇,
                                        p₈,  p₉,  p₁₀, p₁₁, p₁₂, p₁₃, p₁₄, 
                                        p₁₅, p₁₆, p₁₇, p₁₈, p₁₉, p₂₀)
            @test hex20.points == SVector(p₁,  p₂,  p₃,  p₄,  p₅,  p₆,  p₇,
                                          p₈,  p₉,  p₁₀, p₁₁, p₁₂, p₁₃, p₁₄, 
                                          p₁₅, p₁₆, p₁₇, p₁₈, p₁₉, p₂₀)
        end
    end
end
