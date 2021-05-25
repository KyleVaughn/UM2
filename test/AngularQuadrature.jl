using MOCNeutronTransport
include("../src/AngularQuadrature.jl")
@testset "AngularQuadrature" begin
    for type in [Float32, Float64, BigFloat]
        @testset "Chebyshev" begin
            # chebyshev_angular_quadrature
            angles, weights = chebyshev_angular_quadrature(1, type)
            @test size(angles, 1) == size(weights, 1) == 1
            @test angles[1] == π/type(4)
            @test weights[1] == type(1)
            @test typeof(angles[1]) == typeof(weights[1]) == typeof(type(1))

            angles, weights = chebyshev_angular_quadrature(2, type)
            @test size(angles, 1) == size(weights, 1) == 2
            @test angles  == Tuple([type(π)/type(8), type(3)*type(π)/type(8)])
            @test weights == Tuple([type(1/2), type(1/2)])

            angles, weights = chebyshev_angular_quadrature(3, type)
            @test size(angles, 1) == size(weights, 1) == 3
            @test angles  == Tuple([type(π)/type(12), type(3)*type(π)/type(12), type(5)*type(π)/type(12)])
            @test weights == Tuple([type(1)/type(3), type(1)/type(3), type(1)/type(3)])
        end

        @testset "Chebyshev-Chebyshev" begin
            # Default data type is Float64
            quadrature = AngularQuadrature("Chebyshev-Chebyshev", 3, 2)
            @test quadrature isa ProductAngularQuadrature
            @test size(quadrature.γ, 1) == size(quadrature.w_γ, 1) == 3
            @test size(quadrature.θ, 1) == size(quadrature.w_θ, 1) == 2 
            @test quadrature.γ   == Tuple([π/12, 3*π/12, 5*π/12])
            @test quadrature.w_γ == Tuple([1/3, 1/3, 1/3])
            @test quadrature.θ   == Tuple([π/8, 3*π/8])
            @test quadrature.w_θ == Tuple([1/2, 1/2])
            
            # Specified data type
            quadrature = AngularQuadrature("Chebyshev-Chebyshev", 3, 2, T=type)
            @test quadrature isa ProductAngularQuadrature
            @test size(quadrature.γ, 1) == size(quadrature.w_γ, 1) == 3
            @test size(quadrature.θ, 1) == size(quadrature.w_θ, 1) == 2
            @test quadrature.γ   == Tuple([type(π)/type(12), 
                                           type(3)*type(π)/type(12), 
                                           type(5)*type(π)/type(12)])
            @test quadrature.w_γ == Tuple([type(1)/type(3), type(1)/type(3), type(1)/type(3)])
            @test quadrature.θ   == Tuple([type(π)/type(8), type(3)*type(π)/type(8)])
            @test quadrature.w_θ == Tuple([type(1)/type(2), type(1)/type(2)])
        end
    end
end
