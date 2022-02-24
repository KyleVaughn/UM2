using MOCNeutronTransport
@testset "Hyperplane2D" begin
    for F in [Float32, Float64, BigFloat]
        plane = Hyperplane(Point2D{F}(1,1), Point2D{F}(2,2))
        @test plane.𝗻̂ ≈ [-sqrt(F(2))/2, sqrt(F(2))/2]
        @test plane.d ≈ 0
    end
end
@testset "Hyperplane3D" begin
    for F in [Float32, Float64, BigFloat]
        plane = Hyperplane(Point3D{F}(0,0,2), Point3D{F}(1,0,2), Point3D{F}(0,1,2))
        @test plane.𝗻̂ ≈ [0,0,1]
        @test plane.d ≈ 2
    end
end
