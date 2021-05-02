using MOCNeutronTransport: Point, distance
@testset "Point - Constructors" begin
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

    # zero
    p = zero(p)
    @test p.x == 0.0
    @test p.y == 0.0    
    @test p.z == 0.0
end

@testset "Point - Operators" begin
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(2.0, 4.0, 6.0)

    # Addition
    p = p1 + p2
    @test p.x == 3.0 
    @test p.y == 6.0 
    @test p.z == 9.0 

    # Subtraction
    p = p1 - p2
    @test p.x == -1.0 
    @test p.y == -2.0 
    @test p.z == -3.0 

    # Multiplication
    p = 4*p1
    @test p.x == 4.0
    @test p.y == 8.0 
    @test p.z == 12.0 

    # Division
    p = p1/4
    @test p.x == 0.25
    @test p.y == 0.5 
    @test p.z == 0.75 
end

@testset "Point - Methods" begin
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(2.0, 4.0, 6.0)

    @test distance(p1, p2) == sqrt(14.0)
end
