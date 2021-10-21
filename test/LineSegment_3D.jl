using MOCNeutronTransport
@testset "LineSegment_3D" begin
    for T in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_3D( T(1) )
            p₂ = Point_3D( T(2) )
            l = LineSegment_3D(p₁,p₂)
            @test l.points[1] == p₁
            @test l.points[2] == p₂
        end
        @testset "Methods" begin
            # interpolation
            p₁ = Point_3D(T, 1, 1, 3)
            p₂ = Point_3D(T, 3, 3, 3)
            l = LineSegment_3D(p₁, p₂)
            @test l(0) == p₁
            @test l(1) == p₂
            p₃ = Point_3D(T, 2, 2, 3)
            @test l(1//2) == p₃
            typeof(l(1//2).x) == typeof(T.((2, 4, 6)))

#            # arc_length
#            p₁ = Point_3D(T, 1, 2, 3)
#            p₂ = Point_3D(T, 2, 4, 6)
#            l = LineSegment_3D(p₁, p₂)
#            @test arc_length(l) == sqrt(T(14))
#            @test typeof(arc_length(l)) == typeof(T(14))
#
#            # intersect
#            # -------------------------------------------
#            # basic intersection
#            l₁ = LineSegment_3D(Point_3D(T, 0,  1), Point_3D(T, 2, -1))
#            l₂ = LineSegment_3D(Point_3D(T, 0, -1), Point_3D(T, 2,  1))
#            bool, p = intersect(l₁, l₂)
#            @test bool
#            @test p == Point_3D(T, 1, 0)
#            @test typeof(p.x) == typeof(SVector(T.((1, 0, 0))))
#
#            # vertex intersection
#            l₂ = LineSegment_3D(Point_3D(T, 0, -1), Point_3D(T, 2, -1))
#            bool, p = l₁ ∩ l₂
#            @test bool
#            @test p == Point_3D(T, 2, -1)
#
#            # basic 3D intersection
#            l₁ = LineSegment_3D(Point_3D(T, 1, 0,  1), Point_3D(T, 1,  0, -1))
#            l₂ = LineSegment_3D(Point_3D(T, 0, 1, -1), Point_3D(T, 2, -1,  1))
#            bool, p = l₁ ∩ l₂
#            @test bool
#            @test p == Point_3D(T, 1, 0, 0)
#
#            # vertical
#            l₁ = LineSegment_3D(Point_3D(T, 0,  1), Point_3D(T, 2,   1))
#            l₂ = LineSegment_3D(Point_3D(T, 1, 10), Point_3D(T, 1, -10))
#            bool, p = intersect(l₁, l₂)
#            @test bool
#            @test p == Point_3D(T, 1, 1)
#
#            # nearly vertical
#            l₁ = LineSegment_3D(Point_3D(T, -1, -100000), Point_3D(T, 1,  100000))
#            l₂ = LineSegment_3D(Point_3D(T, -1,   10000), Point_3D(T, 1,  -10000))
#            bool, p = l₁ ∩ l₂
#            @test bool
#            @test p == Point_3D(T, 0, 0)
#
#            # parallel
#            l₁ = LineSegment_3D(Point_3D(T, 0, 1), Point_3D(T, 1, 1))
#            l₂ = LineSegment_3D(Point_3D(T, 0, 0), Point_3D(T, 1, 0))
#            bool, p = intersect(l₁, l₂)
#            @test !bool
#
#            # collinear
#            l₁ = LineSegment_3D(Point_3D(T, 0, 0), Point_3D(T, 2, 0))
#            l₂ = LineSegment_3D(Point_3D(T, 0, 0), Point_3D(T, 1, 0))
#            bool, p = intersect(l₁, l₂)
#            @test !bool
#
#            # line intersects, not segment (invalid t)
#            l₁ = LineSegment_3D(Point_3D(T, 0, 0), Point_3D(T, 2, 0    ))
#            l₂ = LineSegment_3D(Point_3D(T, 1, 2), Point_3D(T, 1, 1//10))
#            bool, p = l₁ ∩ l₂
#            @test !bool
#            @test p ≈ Point_3D(T, 1, 0) # the closest point on line 1
#
#            # line intersects, not segment (invalid s)
#            l₂ = LineSegment_3D(Point_3D(T, 0, 0), Point_3D(T, 2, 0    ))
#            l₁ = LineSegment_3D(Point_3D(T, 1, 2), Point_3D(T, 1, 1//10))
#            bool, p = intersect(l₁, l₂)
#            @test !bool
#            @test p ≈ Point_3D(T, 1, 0) # the closest point on line 1
        end
    end
end
