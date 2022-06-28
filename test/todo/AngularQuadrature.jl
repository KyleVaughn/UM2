using MOCNeutronTransport
include("../src/AngularQuadrature.jl")
@testset "AngularQuadrature" begin for type in [Float32, Float64]
    @testset "Chebyshev" begin
        # chebyshev_angular_quadrature
        angles, weights = generate_chebyshev_angular_quadrature(1, type)
        @test size(angles, 1) == size(weights, 1) == 1
        @test angles[1] == π / type(4)
        @test weights[1] == type(1)
        @test typeof(angles[1]) == typeof(weights[1]) == typeof(type(1))

        angles, weights = generate_chebyshev_angular_quadrature(2, type)
        @test size(angles, 1) == size(weights, 1) == 2
        @test angles == type[3π / 8, π / 8]
        @test weights == type[1 // 2, 1 // 2]

        angles, weights = generate_chebyshev_angular_quadrature(3, type)
        @test size(angles, 1) == size(weights, 1) == 3
        @test angles == type[5π / 12, 3π / 12, π / 12]
        @test weights == type[1 / 3, 1 / 3, 1 / 3]
    end

    @testset "Chebyshev-Chebyshev" begin
        # Default data type is Float64
        quadrature = generate_angular_quadrature("Chebyshev-Chebyshev", 3, 2)
        @test quadrature isa ProductAngularQuadrature
        @test size(quadrature.γ, 1) == size(quadrature.w_γ, 1) == 6
        @test size(quadrature.θ, 1) == size(quadrature.w_θ, 1) == 2
        @test quadrature.γ == Tuple([5π / 12, 3π / 12, π / 12, 7π / 12, 9π / 12, 11π / 12])
        @test quadrature.w_γ == Tuple([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        @test quadrature.θ == Tuple([3π / 8, π / 8])
        @test quadrature.w_θ == Tuple([1 / 2, 1 / 2])

        # Specified data type
        quadrature = generate_angular_quadrature("Chebyshev-Chebyshev", 3, 2, T = type)
        @test quadrature isa ProductAngularQuadrature
        @test size(quadrature.γ, 1) == size(quadrature.w_γ, 1) == 6
        @test size(quadrature.θ, 1) == size(quadrature.w_θ, 1) == 2
        γ_ref = Tuple(type[5π / 12, 3π / 12, π / 12, 7π / 12, 9π / 12, 11π / 12])
        for i in 1:6
            @test quadrature.γ[i] ≈ γ_ref[i]
        end
        @test quadrature.w_γ == Tuple(type[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        @test quadrature.θ == Tuple(type[3π / 8, π / 8])
        @test quadrature.w_θ == Tuple(type[1 / 2, 1 / 2])
    end
end end
