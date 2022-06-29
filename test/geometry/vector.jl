@testset "Vec" begin for T in Floats
    a = Vec{3, T}(1, 2, 3)
    b = Vec{3, T}(2, 4, 6)
    @test a ⊙ b ≈ [2, 8, 18]
    if T != BigFloat
        @test @ballocated($a ⊙ $b, samples = 1, evals = 2) == 0
    end

    @test b ⊘ a ≈ [2, 2, 2]
    if T != BigFloat
        @test @ballocated($a ⊘ $b, samples = 1, evals = 2) == 0
    end

    @test inv(a) * a ≈ 1
    if T != BigFloat
        @test @ballocated(inv($a), samples = 1, evals = 2) == 0
    end

    @test norm²(a) ≈ 1 + 4 + 9
    if T != BigFloat
        @test @ballocated(norm²($a), samples = 1, evals = 2) == 0
    end
end end
