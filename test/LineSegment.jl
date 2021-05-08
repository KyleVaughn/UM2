using MOCNeutronTransport
@testset "LineSegment" begin
    # Constructors
    # ---------------------------------------------------------------------------------------------
    p⃗₁ = Point(1.0, 0.0) 
    p⃗₂ = Point(2.0, 0.0)
    l = LineSegment(p⃗₁,p⃗₂)
    @test l.p⃗₁ == p⃗₁
    @test l.p⃗₂ == p⃗₂    

    # Methods
    # ---------------------------------------------------------------------------------------------
    # segment_length
    p⃗₁ = Point(1.0, 2.0, 3.0)
    p⃗₂ = Point(2.0, 4.0, 6.0)
    l = LineSegment(p⃗₁, p⃗₂)
    @test arc_length(l) == sqrt(14.0)

    # evaluation
    p⃗₁ = Point(1.0, 1.0, 3.0)   
    p⃗₂ = Point(3.0, 3.0, 3.0)   
    l = LineSegment(p⃗₁, p⃗₂)            
    @test l(0.0) == p⃗₁
    @test l(1.0) == p⃗₂
    p⃗₃ = Point(2.0, 2.0, 3.0)
    @test l(0.5) == p⃗₃

    # midpoint
    @test midpoint(l) == p⃗₃

    # intersects
    # -------------------------------------------
    # basic intersection
    l₁ = LineSegment(Point(0.0, 1.0), Point(2.0, -1.0))
    l₂ = LineSegment(Point(0.0, -1.0), Point(2.0, 1.0))
    bool, p = intersects(l₁, l₂)
    @test bool
    @test p == Point(1.0, 0.0)

    # vertex intersection
    l₂ = LineSegment(Point(0.0, -1.0), Point(2.0, -1.0))
    bool, p = intersects(l₁, l₂)
    @test bool
    @test p == Point(2.0, -1.0)

    # vertical
    l₁ = LineSegment(Point(0.0, 1.0), Point(2.0, 1.0))
    l₂ = LineSegment(Point(1.0, 10.0), Point(1.0, -10.0))
    bool, p = intersects(l₁, l₂)
    @test bool
    @test p == Point(1.0, 1.0)

    # nearly vertical
    l₁ = LineSegment(Point(-1.0, -100000.0), Point(1.0, 100000.0))
    l₂ = LineSegment(Point(-1.0,   10000.0), Point(1.0,  -10000.0))
    bool, p = intersects(l₁, l₂)
    @test bool
    @test p == Point(0.0, 0.0)

    # parallel
    l₁ = LineSegment(Point(0.0, 1.0), Point(1.0, 1.0))
    l₂ = LineSegment(Point(0.0, 0.0), Point(1.0, 0.0))
    bool, p = intersects(l₁, l₂)
    @test !bool
    @test p == Point(0.0, 0.0)

    # collinear
    l₁ = LineSegment(Point(0.0, 0.0), Point(2.0, 0.0))
    l₂ = LineSegment(Point(0.0, 0.0), Point(1.0, 0.0))
    bool, p = intersects(l₁, l₂)
    @test !bool
    @test p == Point(0.0, 0.0)

    # line intersects, not segment (invalid t)
    l₁ = LineSegment(Point(0.0, 0.0), Point(2.0, 0.0))
    l₂ = LineSegment(Point(1.0, 2.0), Point(1.0, 0.1))
    bool, p = intersects(l₁, l₂)
    @test !bool
    @test p ≈ Point(1.0, 0.0) # the closest point on line 1

    # line intersects, not segment (invalid s)
    l₂ = LineSegment(Point(0.0, 0.0), Point(2.0, 0.0))
    l₁ = LineSegment(Point(1.0, 2.0), Point(1.0, 0.1))
    bool, p = intersects(l₁, l₂)
    @test !bool
    @test p ≈ Point(1.0, 0.0) # the closest point on line 1

    # isleft
    # -------------------------------------------
    l = LineSegment(Point(0.0, 0.0), Point(0.0, 1.0))
    @test is_left(Point(-1.0, 0.0), l)
    @test is_left(Point(-0.001, -10.0), l)
    @test !is_left(Point(1.0, 0.0), l)
    @test !is_left(Point(0.001, -10.0), l)
end
