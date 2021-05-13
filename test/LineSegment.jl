using MOCNeutronTransport
@testset "LineSegment" begin
    for type in [Float32, Float64, BigFloat]
        # Constructors
        # ---------------------------------------------------------------------------------------------
        p⃗₁ = Point( type.((1, 0)) )
        p⃗₂ = Point( type.((2, 0)) )
        l = LineSegment(p⃗₁,p⃗₂)
        @test l.p⃗₁ == p⃗₁
        @test l.p⃗₂ == p⃗₂

        # Methods
        # ---------------------------------------------------------------------------------------------
        # arc_length
        p⃗₁ = Point( type.((1, 2, 3)) )
        p⃗₂ = Point( type.((2, 4, 6)) )
        l = LineSegment(p⃗₁, p⃗₂)
        @test arc_length(l) == sqrt(type(14))
        @test typeof(arc_length(l)) == typeof(type(14))

        # evaluation
        p⃗₁ = Point( type.((1, 1, 3)) )
        p⃗₂ = Point( type.((3, 3, 3)) )
        l = LineSegment(p⃗₁, p⃗₂)
        @test l(0.0) == p⃗₁
        @test l(1.0) == p⃗₂
        p⃗₃ = Point( type.((2, 2, 3)) )
        @test l(0.5) == p⃗₃
        typeof(l(0.5).coord) == typeof(type.((2, 4, 6)))

        # midpoint
        @test midpoint(l) == p⃗₃

        # intersects
        # -------------------------------------------
        # basic intersection
        l₁ = LineSegment(Point( type.((0, 1)) ), Point( type.((2, -1)) ))
        l₂ = LineSegment(Point( type.((0, -1.)) ), Point( type.((2, 1)) ))
        bool, p = intersects(l₁, l₂)
        @test bool
        @test p == Point(type.((1, 0)))
        @test typeof(p.coord) == typeof(type.((1, 0, 0)))

        # vertex intersection
        l₂ = LineSegment(Point( type.((0, -1)) ), Point( type.((2, -1)) ))
        bool, p = intersects(l₁, l₂)
        @test bool
        @test p == Point(2.0, -1.0)

        # basic 3D intersection
        l₁ = LineSegment(Point( type.((1, 0,  1)) ), Point( type.((1,  0, -1)) ))
        l₂ = LineSegment(Point( type.((0, 1, -1)) ), Point( type.((2, -1,  1)) ))
        bool, p = intersects(l₁, l₂)
        @test bool
        @test p == Point(1.0, 0.0, 0.0)

        # vertical
        l₁ = LineSegment(Point( type.((0,  1)) ), Point(type.((2,   1))))
        l₂ = LineSegment(Point( type.((1, 10)) ), Point(type.((1, -10))))
        bool, p = intersects(l₁, l₂)
        @test bool
        @test p == Point(1.0, 1.0)

        # nearly vertical
        l₁ = LineSegment(Point( type.((-1, -100000)) ), Point( type.((1,  100000)) ))
        l₂ = LineSegment(Point( type.((-1,   10000)) ), Point( type.((1,  -10000)) ))
        bool, p = intersects(l₁, l₂)
        @test bool
        @test p == Point(0.0, 0.0)

        # parallel
        l₁ = LineSegment(Point( type.((0, 1)) ), Point(type.((1, 1))))
        l₂ = LineSegment(Point( type.((0, 0)) ), Point(type.((1, 0))))
        bool, p = intersects(l₁, l₂)
        @test !bool
        @test p == Point(Inf, Inf, Inf)

        # collinear
        l₁ = LineSegment(Point( type.((0, 0)) ), Point( type.((2, 0)) ))
        l₂ = LineSegment(Point( type.((0, 0)) ), Point( type.((1, 0)) ))
        bool, p = intersects(l₁, l₂)
        @test !bool
        @test p == Point(Inf, Inf, Inf)

        # line intersects, not segment (invalid t)
        l₁ = LineSegment(Point( type.((0, 0)) ), Point(type.((2, 0)) ))
        l₂ = LineSegment(Point( type.((1, 2)) ), Point(type.((1, 0.1)) ))
        bool, p = intersects(l₁, l₂)
        @test !bool
        @test p ≈ Point(type.((1, 0))) # the closest point on line 1

        # line intersects, not segment (invalid s)
        l₂ = LineSegment(Point( type.((0, 0)) ), Point( type.((2, 0)) ))
        l₁ = LineSegment(Point( type.((1, 2)) ), Point( type.((1, 0.1)) ))
        bool, p = intersects(l₁, l₂)
        @test !bool
        @test p ≈ Point(type.((1, 0))) # the closest point on line 1

        # isleft
        # -------------------------------------------
        l = LineSegment(Point( type.((0, 0)) ), Point( type.((0, 1))))
        @test  is_left(Point(type(-1    ), type(0)    ), l)
        @test  is_left(Point(type(-0.001), type(-10.0)), l)
        @test !is_left(Point(type(1.0   ), type(0.0)  ), l)
        @test !is_left(Point(type(0.001 ), type(-10.0)), l)
    end
end
