@testset "AABox" begin
    @testset "AABox2D" begin
        for T in [Float32, Float64, BigFloat]
            # getproperty
            aab = AABox2D(Point2D{T}(1, 0), Point2D{T}(3, 2))
            @test aab.xmin ≈ 1
            @test aab.ymin ≈ 0
            @test aab.xmax ≈ 3
            @test aab.ymax ≈ 2
    
            # Δx, Δy
            @test Δx(aab) ≈ 2
            @test Δy(aab) ≈ 2
        end
    end
    @testset "AABox3D" begin
        for T in [Float32, Float64, BigFloat]
            # getproperty
            aab = AABox3D(Point3D{T}(1, 0, -1), Point3D{T}(3, 2, 1))
            @test aab.xmin ≈ 1
            @test aab.ymin ≈ 0
            @test aab.zmin ≈ -1
            @test aab.xmax ≈ 3
            @test aab.ymax ≈ 2
            @test aab.zmax ≈ 1
    
            # Δx, Δy, Δz
            @test Δx(aab) ≈ 2
            @test Δy(aab) ≈ 2
            @test Δz(aab) ≈ 2
        end
    end
end
