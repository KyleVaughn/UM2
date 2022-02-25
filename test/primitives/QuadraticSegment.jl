@testset "QuadraticSegment" begin
    @testset "QuadraticSegment2D" begin
        for T in [Float32, Float64, BigFloat]
            # Constructor
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test q.points == SVector(𝘅₁, 𝘅₂, 𝘅₃)
    
            # isstraight
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 0)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test isstraight(q)
            𝘅₂ = Point2D{T}(2, 0.0001)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test !isstraight(q)
        end
    end

    @testset "QuadraticSegment3D" begin
        for T in [Float32, Float64, BigFloat]
            # Constructor
            𝘅₁ = Point3D{T}(0, 0, 0)
            𝘅₂ = Point3D{T}(0, 2, 0)
            𝘅₃ = Point3D{T}(0, 1, 1)
            q = QuadraticSegment3D(𝘅₁, 𝘅₂, 𝘅₃)
            @test q.points == SVector(𝘅₁, 𝘅₂, 𝘅₃)
    
            # isstraight
            𝘅₁ = Point3D{T}(0, 0, 0)
            𝘅₂ = Point3D{T}(0, 2, 0)
            𝘅₃ = Point3D{T}(0, 1, 0)
            q = QuadraticSegment3D(𝘅₁, 𝘅₂, 𝘅₃)
            @test isstraight(q)
            𝘅₂ = Point3D{T}(0, 2, 0.0001)
            q = QuadraticSegment3D(𝘅₁, 𝘅₂, 𝘅₃)
            @test !isstraight(q)
        end
    end
end
