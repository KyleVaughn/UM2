using MOCNeutronTransport
@testset "Point" begin
    # Constructors
    # ---------------------------------------------------------------------------------------------
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
    
    # Operators
    # ---------------------------------------------------------------------------------------------
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(2.0, 4.0, 6.0)

    # Point addition
    p = p1 + p2
    @test p.x == 3.0 
    @test p.y == 6.0 
    @test p.z == 9.0 

    # Point subtraction
    p = p1 - p2
    @test p.x == -1.0 
    @test p.y == -2.0 
    @test p.z == -3.0 
    
    # Addition
    p = p1 + 1.0
    @test p.x == 2.0 
    @test p.y == 3.0 
    @test p.z == 4.0 

    # Subtraction
    p = p1 - 1.0
    @test p.x == 0.0 
    @test p.y == 1.0 
    @test p.z == 2.0 

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

    # Methods
    # ---------------------------------------------------------------------------------------------
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(2.0, 4.0, 6.0)
    # distance
    @test distance(p1, p2) == sqrt(14.0)
end
