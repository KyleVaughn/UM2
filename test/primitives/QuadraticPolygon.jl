@testset "QuadraticPolygon" begin
    @testset "QuadraticTriangle2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(1, 1)
            p₄ = Point2D{T}(1//2, 0)
            p₅ = Point2D{T}(1, 1//2)
            p₆ = Point2D{T}(1//2, 1//2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆)
        end
    end
    
    @testset "QuadraticQuadrilateral2D" begin
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
            @test quad8.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end
    end

    @testset "QuadraticTriangle3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 1, 1)
            p₄ = Point3D{T}(0, 1//2, 0)
            p₅ = Point3D{T}(0, 1, 1//2)
            p₆ = Point3D{T}(0, 1//2, 1//2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆)
        end
    end
    
    @testset "QuadraticQuadrilateral3D" begin
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
            @test quad8.points == SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
        end
    end
end
