using MOCNeutronTransport
@testset "Line" begin
    # Constructors
    # ---------------------------------------------------------------------------------------------
    p1 = Point(1.0) 
    p2 = Point(2.0)
    l = Line(p1,p2)
    @test l.p1 == p1
    @test l.p2 == p2    

    # Methods
    # ---------------------------------------------------------------------------------------------
    # distance
    p1 = Point(1.0, 2.0, 3.0)
    p2 = Point(2.0, 4.0, 6.0)
    l = Line(p1, p2)
    @test distance(l) == sqrt(14.0)

    # evaluation
    p1 = Point(1.0, 1.0, 3.0)   
    p2 = Point(3.0, 3.0, 3.0)   
    l = Line(p1, p2)            
    @test l(0.0) == p1
    @test l(1.0) == p2
    p3 = Point(2.0, 2.0, 3.0)
    @test l(0.5) == p3
end
