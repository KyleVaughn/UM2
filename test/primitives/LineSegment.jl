@testset "LineSegment" begin
    @testset "LineSegment2D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point2D{T}(1, 0)
            p₂ = Point2D{T}(2, 0)
            l = LineSegment2D(p₁, p₂)
            @test l.𝘅₁== p₁
            @test l.𝘂 == p₂ - p₁
    
            # interpolation
            p₁ = Point2D{T}(1, 1)
            p₂ = Point2D{T}(3, 3)
            l = LineSegment2D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point2D{T}(2, 2)
        end
    end
    
    @testset "LineSegment3D" begin
        for T in [Float32, Float64, BigFloat]
            p₁ = Point3D{T}(1, 0, 1)
            p₂ = Point3D{T}(2, 0, -1)
            l = LineSegment3D(p₁, p₂)
            @test l.𝘅₁== p₁
            @test l.𝘂 == p₂ - p₁
    
            # interpolation
            p₁ = Point3D{T}(1, 1, 1)
            p₂ = Point3D{T}(3, 3, 3)
            l = LineSegment3D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point3D{T}(2, 2, 2)
        end
    end
end
