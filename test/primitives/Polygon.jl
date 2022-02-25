@testset "Polygon" begin
    @testset "Triangle2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
        end
    end
    
    @testset "Triangle3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
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
        end
    end

    @testset "Quadrilateral3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 1, 1)
            p₄ = Point3D{T}(0, 0, 1)
            quad = Quadrilateral(p₁, p₂, p₃, p₄)
            @test quad.points == SVector(p₁, p₂, p₃, p₄)
        end
    end
end
