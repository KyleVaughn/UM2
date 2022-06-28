@testset "AABox" begin
    @testset "AABox{2}" begin for T in Floats
        P₁ = Point{2, T}(1, 0)
        P₂ = Point{2, T}(3, 2)

        # getproperty
        aab = AABox(P₁, P₂)
        @test xmin(aab) ≈ 1
        @test ymin(aab) ≈ 0
        @test xmax(aab) ≈ 3
        @test ymax(aab) ≈ 2

        # Δx, Δy
        @test Δx(aab) ≈ 2
        @test Δy(aab) ≈ 2

        # measure
        @test measure(aab) ≈ 4
    end end
    @testset "AABox{3}" begin for T in Floats
        # getproperty
        aab = AABox(Point{3, T}(1, 0, -1), Point{3, T}(3, 2, 1))
        @test xmin(aab) ≈ 1
        @test ymin(aab) ≈ 0
        @test zmin(aab) ≈ -1
        @test xmax(aab) ≈ 3
        @test ymax(aab) ≈ 2
        @test zmax(aab) ≈ 1

        # Δx, Δy, Δz
        @test Δx(aab) ≈ 2
        @test Δy(aab) ≈ 2
        @test Δz(aab) ≈ 2

        # measure
        @test measure(aab) ≈ 8
    end end
end
