using MOCNeutronTransport: Point, distance
@testset "Point" begin
    # Constructors
    # 3D
    p = Point(1.0, 2.0, 3.0)
    @test p.x == 1.0
    @test p.y == 2.0
    @test p.z == 3.0

    # 2D
    p = Point(1.0, 2.0)
    @test p.x == 1.0
    @test p.y == 2.0
    @test p.z == 0.0

    # 1D
    p = Point(1.0) 
    @test p.x == 1.0
    @test p.y == 0.0    
    @test p.z == 0.0

    # Distance
    p1 = Point(1.0)
    p2 = Point(1.0)
    @test distance(p1, p2) == 0.0
end
