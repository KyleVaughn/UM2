@testset "Rectilinear Grid" begin
    @testset "RectilinearGrid2D" begin
        for T in Floats
            bb = AABox2D(Point2D{T}(0, 0), Point2D{T}(1, 1))
            xdiv = SVector{3, T}(1//4, 1//2, 3//4)
            ydiv = SVector{2, T}(1//3, 2//3)
            rg = RectilinearGrid2D(bb, xdiv, ydiv)

            @test rg.bb == bb
            @test rg.xdiv == xdiv
            @test rg.ydiv == ydiv
        end
    end
end
