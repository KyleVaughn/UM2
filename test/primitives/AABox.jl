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

            # split
            aab = AABox2D(Point2D{T}(0,0), Point2D{T}(3, 2))
            xdiv = SVector{2, T}(2,1)
            ydiv = SVector{1, T}(1)
            aabs = split(aab, xdiv, ydiv)
            @test size(aabs) == (3,2)
            @test aabs[1].minima ≈ Point2D{T}(0, 0)
            @test aabs[1].maxima ≈ Point2D{T}(1, 1)
            @test aabs[2].minima ≈ Point2D{T}(1, 0)
            @test aabs[2].maxima ≈ Point2D{T}(2, 1)
            @test aabs[3].minima ≈ Point2D{T}(2, 0)
            @test aabs[3].maxima ≈ Point2D{T}(3, 1)
            @test aabs[4].minima ≈ Point2D{T}(0, 1)
            @test aabs[4].maxima ≈ Point2D{T}(1, 2)
            @test aabs[5].minima ≈ Point2D{T}(1, 1)
            @test aabs[5].maxima ≈ Point2D{T}(2, 2)
            @test aabs[6].minima ≈ Point2D{T}(2, 1)
            @test aabs[6].maxima ≈ Point2D{T}(3, 2)

            xdiv = SVector{2, T}(2,1)
            ydiv = SVector{0, T}()
            aabs = split(aab, xdiv, ydiv)
            @test size(aabs) == (3,1)
            @test aabs[1].minima ≈ Point2D{T}(0, 0)
            @test aabs[1].maxima ≈ Point2D{T}(1, 2)
            @test aabs[2].minima ≈ Point2D{T}(1, 0)
            @test aabs[2].maxima ≈ Point2D{T}(2, 2)
            @test aabs[3].minima ≈ Point2D{T}(2, 0)
            @test aabs[3].maxima ≈ Point2D{T}(3, 2)

            xdiv = SVector{0, T}()
            ydiv = SVector{1, T}(1)
            aabs = split(aab, xdiv, ydiv)
            @test size(aabs) == (1,2)
            @test aabs[1].minima ≈ Point2D{T}(0, 0)
            @test aabs[1].maxima ≈ Point2D{T}(3, 1)
            @test aabs[2].minima ≈ Point2D{T}(0, 1)
            @test aabs[2].maxima ≈ Point2D{T}(3, 2)
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
