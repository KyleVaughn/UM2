@testset "Select Type" begin @testset "UInt" begin
    @test UM2._select_uint_type(2^0) == UInt8
    @test UM2._select_uint_type(2^8) == UInt16
    @test UM2._select_uint_type(2^16) == UInt32
    @test UM2._select_uint_type(2^32) == UInt64

    @test @allocated(UM2._select_uint_type(2^64)) == 0
end end
