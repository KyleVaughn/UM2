@testset "LineSegment" begin
    @testset "LineSegment2D" begin
        for T ∈ Floats
            p₁ = Point2D{T}(1, 0)
            p₂ = Point2D{T}(2, 0)
            l = LineSegment2D(p₁, p₂)
            @test l.𝘅₁== p₁
            @test l.𝘂 == p₂ - p₁
        end
    end
    
    @testset "LineSegment3D" begin
        for T ∈ Floats
            p₁ = Point3D{T}(1, 0, 1)
            p₂ = Point3D{T}(2, 0, -1)
            l = LineSegment3D(p₁, p₂)
            @test l.𝘅₁== p₁
            @test l.𝘂 == p₂ - p₁
        end
    end
end
