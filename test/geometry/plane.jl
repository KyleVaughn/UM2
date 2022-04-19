@testset "Plane" begin
    @testset "Plane" begin
        for T ∈ Points3D
            plane = Plane(T(0,0,2), T(1,0,2), T(0,1,2))
            @test plane.𝗻̂ ≈ [0,0,1]
            @test plane.d ≈ 2
        end
    end
end
