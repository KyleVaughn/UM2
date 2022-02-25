@testset "Polygon" begin
    @testset "Triangle2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
    
            # interpolation
            @test tri(0, 0) ≈ p₁
            @test tri(1, 0) ≈ p₂
            @test tri(0, 1) ≈ p₃
            @test tri(1//2, 1//2) ≈ Point2D{T}(1//2, 1//2)
        end
    end
    
    @testset "Triangle3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
    
            # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point3D{T}(0, 1//2, 1//2)
        end
    end
    
    @testset "Quadrilateral2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(1, 1)
            p₄ = Point2D{T}(0, 1)
            quad = Quadrilateral(p₁, p₂, p₃, p₄)
            @test quad.points == SVector(p₁, p₂, p₃, p₄)
    
            # interpolation
            @test quad(0, 0) ≈ p₁
            @test quad(1, 0) ≈ p₂
            @test quad(1, 1) ≈ p₃
            @test quad(0, 1) ≈ p₄
            @test quad(1//2, 1//2) ≈ Point2D{T}(1//2, 1//2)
        end
    end
end
