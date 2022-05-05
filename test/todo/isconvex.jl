@testset "Polygon" begin
    @testset "Triangle2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)

            @test isconvex(tri)
        end
    end
    
    @testset "Triangle3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(0, 0, 0)
            p₂ = Point3D{T}(0, 1, 0)
            p₃ = Point3D{T}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
            
            @test isplanar(tri)
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

            @test isconvex(quad)
            @test !isconvex(Quadrilateral(p₄, p₃, p₂, p₁))
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

            @test isplanar(quad)
            @test !isplanar(Quadrilateral(p₁, p₂, p₃, Point3D{T}(1//10, 0, 1)))
        end
    end
end
