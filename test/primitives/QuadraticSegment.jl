@testset "QuadraticSegment" begin
    @testset "QuadraticSegment2D" begin
        for T in [Float32, Float64, BigFloat]
            # Constructor
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test q.points == SVector(𝘅₁, 𝘅₂, 𝘅₃)
    
            # interpolation
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            for r = LinRange{T}(0, 1, 11)
                @test q(r) ≈ Point2D{T}(2r, -(2r)^2 + 4r)
            end
     
            # jacobian 
            for r = LinRange{T}(0, 1, 11)
                @test 𝗝(q, r) ≈ SVector{2,T}(2, -(8r) + 4)
            end
     
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
end
