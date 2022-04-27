@testset "Rectilinear Grid" begin
    @testset "Rectilinear Grid - 2D" begin
        for T in Floats
            x = Vec{5,T}(0, 1//4, 1//2, 3//4, 1)
            y = Vec{4,T}(0, 1//3, 2//3, 1)
            rg = RectilinearGrid(x, y)
            @test rg.x == x
            @test rg.y == y
        end
    end

    @testset "Rectilinear Grid - 3D" begin
        for T in Floats
            x = Vec{5,T}(0, 1//4, 1//2, 3//4, 1)
            y = Vec{4,T}(0, 1//3, 2//3, 1)
            z = Vec{4,T}(0, 1//3, 2//3, 1)
            rg = RectilinearGrid(x, y, z)
            @test rg.x == x
            @test rg.y == y
            @test rg.z == z
        end
    end
end
