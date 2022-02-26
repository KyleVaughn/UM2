@testset "Interpolation" begin
    @testset "LineSegment" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(1, 1, 1)
            p₂ = Point3D{T}(3, 3, 3)
            l = LineSegment(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point3D{T}(2, 2, 2)
        end
    end

    @testset "QuadraticSegment" begin
        for T in [Float32, Float64, BigFloat]
            𝘅₁ = Point3D{T}(0, 0, 0)
            𝘅₂ = Point3D{T}(2, 0, 0)
            𝘅₃ = Point3D{T}(1, 1, 0)
            q = QuadraticSegment(𝘅₁, 𝘅₂, 𝘅₃)
            for r = LinRange{T}(0, 1, 5)
                @test q(r) ≈ Point3D{T}(2r, -(2r)^2 + 4r, 0)
            end
        end
    end

    @testset "Triangle" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri(0, 0) ≈ p₁
            @test tri(1, 0) ≈ p₂
            @test tri(0, 1) ≈ p₃
            @test tri(1//2, 1//2) ≈ Point3D{T}(0, 1//2, 1//2)
        end
    end

    @testset "Quadrilateral" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 1, 1)
            p₄ = Point3D{T}(0, 0, 1)
            quad = Quadrilateral(p₁, p₂, p₃, p₄)
            @test quad(0, 0) ≈ p₁
            @test quad(1, 0) ≈ p₂
            @test quad(1, 1) ≈ p₃
            @test quad(0, 1) ≈ p₄
            @test quad(1//2, 1//2) ≈ Point3D{T}(0, 1//2, 1//2)
        end
    end

    @testset "QuadraticTriangle" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 1, 1)
            p₄ = Point3D{T}(0, 1//2, 0)
            p₅ = Point3D{T}(0, 1, 1//2)
            p₆ = Point3D{T}(0, 1//2, 1//2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6(0, 0) ≈ p₁
            @test tri6(1, 0) ≈ p₂
            @test tri6(0, 1) ≈ p₃
            @test tri6(1//2, 0) ≈ p₄
            @test tri6(1//2, 1//2) ≈ p₅
            @test tri6(0, 1//2) ≈ p₆
        end
    end

    @testset "QuadraticQuadrilateral" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 1, 1)
            p₄ = Point3D{T}(0, 0, 1)
            p₅ = Point3D{T}(0, 1//2,    0)
            p₆ = Point3D{T}(0,    1, 1//2)
            p₇ = Point3D{T}(0, 1//2,    1)
            p₈ = Point3D{T}(0,    0, 1//2)
            quad8 = QuadraticQuadrilateral(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            @test quad8(0, 0) ≈ p₁
            @test quad8(1, 0) ≈ p₂
            @test quad8(1, 1) ≈ p₃
            @test quad8(0, 1) ≈ p₄
            @test quad8(1//2,    0) ≈ p₅
            @test quad8(   1, 1//2) ≈ p₆
            @test quad8(1//2,    1) ≈ p₇
            @test quad8(   0, 1//2) ≈ p₈
            @test quad8(1//2, 1//2) ≈ Point3D{T}(0, 1//2, 1//2)
        end
    end

    @testset "Tetrahedron" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)           
            p₂ = Point3D{T}(1, 0, 0)
            p₃ = Point3D{T}(0, 1, 0)
            p₄ = Point3D{T}(0, 0, 1)
            tet = Tetrahedron(p₁, p₂, p₃, p₄)
            @test tet(0, 0, 0) ≈ p₁
            @test tet(1, 0, 0) ≈ p₂
            @test tet(0, 1, 0) ≈ p₃
            @test tet(0, 0, 1) ≈ p₄
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
            @test hex(0, 0, 0) ≈ p₁
            @test hex(1, 0, 0) ≈ p₂
            @test hex(1, 1, 0) ≈ p₃
            @test hex(0, 1, 0) ≈ p₄
            @test hex(0, 0, 1) ≈ p₅
            @test hex(1, 0, 1) ≈ p₆
            @test hex(1, 1, 1) ≈ p₇
            @test hex(0, 1, 1) ≈ p₈
        end
    end






end
