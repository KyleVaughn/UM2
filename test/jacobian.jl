@testset "Jacobian" begin
    @testset "QuadraticSegment" begin
        for T in [Float32, Float64, BigFloat]
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            for r = LinRange{T}(0, 1, 11) 
                @test 𝗝(q, r) ≈ SVector{2,T}(2, -(8r) + 4)
            end
        end
    end

    @testset "QuadraticTriangle" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(0, 1)
            p₄ = Point2D{T}(1//2, 0)
            p₅ = Point2D{T}(1//2, 1//2)
            p₆ = Point2D{T}(0, 1//2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            J = jacobian(tri6, 0, 0)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
            J = jacobian(tri6, 1, 0)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
            J = jacobian(tri6, 0, 1)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
            J = jacobian(tri6, 1//2, 1//2)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
        end
    end

    @testset "QuadraticTriangle" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(1, 1)
            p₄ = Point2D{T}(0, 1)
            p₅ = Point2D{T}(1//2,    0)  
            p₆ = Point2D{T}(   1, 1//2)
            p₇ = Point2D{T}(1//2,    1)  
            p₈ = Point2D{T}(   0, 1//2)
            quad8 = QuadraticQuadrilateral(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
            J = jacobian(quad8, 0, 0)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
            J = jacobian(quad8, 1, 0)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
            J = jacobian(quad8, 1, 1)
            @test J[1] ≈ 1
            @test abs(J[2]) < 1e-6
            @test abs(J[3]) < 1e-6
            @test J[4] ≈ 1
        end
    end

    # TODO: quadratic tetrahedron, quadratic hexahedron

end
