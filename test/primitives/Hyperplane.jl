@testset "Hyperplane" begin
    @testset "Hyperplane2D" begin
        for T in [Float32, Float64, BigFloat]
            plane = Hyperplane(Point2D{T}(1,1), Point2D{T}(2,2))
            @test plane.𝗻̂ ≈ [-sqrt(T(2))/2, sqrt(T(2))/2]
            @test plane.d ≈ 0
        end
    end
    @testset "Hyperplane3D" begin
        for T in [Float32, Float64, BigFloat]
            plane = Hyperplane(Point3D{T}(0,0,2), Point3D{T}(1,0,2), Point3D{T}(0,1,2))
            @test plane.𝗻̂ ≈ [0,0,1]
            @test plane.d ≈ 2
        end
    end
end
