@testset "ConvexPolyhedron" begin
    @testset "Tetrahedron" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(1, 0, 0)
            p₃ = Point3D{T}(0, 1, 0)
            p₄ = Point3D{T}(0, 0, 1)
            tet = Tetrahedron(p₁, p₂, p₃, p₄)
            @test tet.points == SVector(p₁, p₂, p₃, p₄)
        end
    end
    
    @testset "Hexahedron" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(1, 0, 0)
            p₃ = Point3D{T}(0, 1, 0)
            p₄ = Point3D{T}(1, 1, 0)
            p₅ = Point3D{T}(0, 0, 1)
            p₆ = Point3D{T}(1, 0, 1)
            p₇ = Point3D{T}(0, 1, 1)
            p₈ = Point3D{T}(1, 1, 1)
            hex = Hexahedron(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            @test hex.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end
    end
end
