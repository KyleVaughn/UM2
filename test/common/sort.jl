@testset "Sort" begin
    @testset "findsortedfirst" begin for T in Floats
        v = T.(collect(1:10))
        @test UM2.findsortedfirst(v, 3.1) == 4
        @test UM2.findsortedfirst(v, 13.1) == 11

        @test @allocated(UM2.findsortedfirst(v, 3.1)) == 0
    end end

    @testset "getsortedfirst" begin for T in Floats
        # findsortedfirst
        v = T.(collect(1:10))
        @test UM2.getsortedfirst(v, 3.1) == 4
        @test UM2.getsortedfirst(v, 13.1) == 11

        # searchsortedfirst
        v = T.(collect(1:100))
        @test UM2.getsortedfirst(v, 3.1) == 4
        @test UM2.getsortedfirst(v, 103.1) == 101

        @test @allocated(UM2.getsortedfirst(v, 3.1)) == 0
    end end
end
