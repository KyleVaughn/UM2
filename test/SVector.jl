using StaticArrays
@testset "SVector" begin
    for T in [Float32, Float64, BigFloat]
        a = SVector{3, T}(1,2,3)
        b = SVector{3, T}(2,4,6)
        @test a * b ≈ [2, 8, 18]
        @test b / a ≈ [2, 2, 2]
        @test norm²(a) ≈ 1 + 4 + 9
        @test normalize(SVector{2,T}(10,10)) ≈ [1/√(T(2)), 1/√(T(2))]
        @test distance(a, b) ≈ sqrt(T(1 + 4 + 9))
        @test inv(a)*a ≈ 1
    end
end
