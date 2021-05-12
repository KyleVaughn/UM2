using MOCNeutronTransport
include("../src/AngularQuadrature.jl")
@testset "AngularQuadrature" begin
    # chebyshev_angular_quadrature
    angles, weights = chebyshev_angular_quadrature(1)
    @test size(angles, 1) == size(weights, 1) == 1
    @test angles[1] == π/4
    @test weights[1] == 1.0

    angles, weights = chebyshev_angular_quadrature(2)
    @test size(angles, 1) == size(weights, 1) == 2
    @test angles == [π/8, 3*π/8]
    @test weights == [1/2, 1/2]

    angles, weights = chebyshev_angular_quadrature(3)
    @test size(angles, 1) == size(weights, 1) == 3
    @test angles == [π/12, 3*π/12, 5*π/12]
    @test weights == [1/3, 1/3, 1/3]

    quadrature = AngularQuadrature("Chebyshev-Chebyshev", 3, 2)
    @test quadrature isa ProductAngularQuadrature
    @test size(quadrature.θ, 1) == size(quadrature.w_θ, 1) == 3
    @test size(quadrature.γ, 1) == size(quadrature.w_γ, 1) == 2
    @test quadrature.θ   == [π/12, 3*π/12, 5*π/12]
    @test quadrature.w_θ == [1/3, 1/3, 1/3]
    @test quadrature.γ   == [π/8, 3*π/8]
    @test quadrature.w_γ == [1/2, 1/2]

end
