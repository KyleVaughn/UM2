@testset "Angular Quadrature" begin @testset "Chebyshev-Chebyshev" begin for T in Floats
    paq = angular_quadrature(:chebyshev, 4, :chebyshev, 1, T)
    @test all(w -> w ≈ 1 // 4, paq.wγ)
    @test paq.γ[1] ≈ π * 1 // 16
    @test paq.γ[2] ≈ π * 3 // 16
    @test paq.γ[3] ≈ π * 5 // 16
    @test paq.γ[4] ≈ π * 7 // 16
    @test all(w -> w ≈ 1, paq.wθ)
    @test paq.θ[1] ≈ π * 1 // 4
end end end
