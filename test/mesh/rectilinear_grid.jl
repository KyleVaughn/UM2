@testset "Rectilinear Grid" begin
    @testset "Rectilinear Grid - 2D" begin
        for T in Floats
            bb = AABox(Point{2,T}(0, 0), Point{2,T}(1, 1))
            xdiv = Vec{3,T}(1//4, 1//2, 3//4)
            ydiv = Vec{2,T}(1//3, 2//3)
            rg = RectilinearGrid(bb, xdiv, ydiv)

            @test rg.bb == bb
            @test rg.xdiv == xdiv
            @test rg.ydiv == ydiv
        end
    end

    @testset "Rectilinear Grid - 3D" begin
        for T in Floats
            bb = AABox(Point{3,T}(0, 0, 0), Point{3,T}(1, 1, 1))
            xdiv = Vec{3,T}(1//4, 1//2, 3//4)
            ydiv = Vec{2,T}(1//3, 2//3)
            zdiv = Vec{2,T}(1//3, 2//3)
            rg = RectilinearGrid(bb, xdiv, ydiv, zdiv)

            @test rg.bb == bb
            @test rg.xdiv == xdiv
            @test rg.ydiv == ydiv
            @test rg.zdiv == zdiv
        end
    end
end
