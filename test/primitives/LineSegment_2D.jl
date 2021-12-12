using MOCNeutronTransport
@testset "LineSegment_2D" begin
    for F in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_2D(F(1))
            p₂ = Point_2D(F(2))
            l = LineSegment_2D(p₁, p₂)
            @test l.points[1] == p₁
            @test l.points[2] == p₂
        end
        @testset "Methods" begin
            # interpolation
            p₁ = Point_2D(F, 1, 1)
            p₂ = Point_2D(F, 3, 3)
            l = LineSegment_2D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            p₃ = Point_2D(F, 2, 2)
            @test l(1//2) ≈ p₃
            l(1//2) == Point_2D{F} 

            # arc_length
            p₁ = Point_2D(F, 1, 2)
            p₂ = Point_2D(F, 2, 4)
            l = LineSegment_2D(p₁, p₂)
            @test arc_length(l) ≈ sqrt(5)
            @test typeof(arc_length(l)) == F 

            # intersect
            # -------------------------------------------
            # basic intersection
            l₁ = LineSegment_2D(Point_2D(F, 0,  1), Point_2D(F, 2, -1))
            l₂ = LineSegment_2D(Point_2D(F, 0, -1), Point_2D(F, 2,  1))
            npoints, p₁ = intersect(l₁, l₂)
            @test npoints == 1
            @test p₁ == Point_2D(F, 1, 0)
            @test typeof(p₁) == Point_2D{F} 

            # vertex intersection
            l₂ = LineSegment_2D(Point_2D(F, 0, -1), Point_2D(F, 2, -1))
            npoints, p₁ = l₁ ∩ l₂
            @test npoints == 1
            @test p₁ ≈ Point_2D(F, 2, -1)

            # vertical
            l₁ = LineSegment_2D(Point_2D(F, 0,  1), Point_2D(F, 2,   1))
            l₂ = LineSegment_2D(Point_2D(F, 1, 10), Point_2D(F, 1, -10))
            npoints, p₁ = intersect(l₁, l₂)
            @test npoints == 1
            @test p₁ ≈ Point_2D(F, 1, 1)

            # nearly vertical
            l₁ = LineSegment_2D(Point_2D(F, -1, -100000), Point_2D(F, 1,  100000))
            l₂ = LineSegment_2D(Point_2D(F, -1,   10000), Point_2D(F, 1,  -10000))
            npoints, p₁ = l₁ ∩ l₂
            @test npoints == 1
            @test p₁ ≈ Point_2D(F, 0, 0)

            # parallel
            l₁ = LineSegment_2D(Point_2D(F, 0, 1), Point_2D(F, 1, 1))
            l₂ = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 1, 0))
            npoints, p₁ = intersect(l₁, l₂)
            @test npoints == 0

            # collinear
            l₁ = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 2, 0))
            l₂ = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 1, 0))
            npoints, p₁ = intersect(l₁, l₂)
            @test npoints == 0

            # line intersects, not segment (invalid t)
            l₁ = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 2, 0    ))
            l₂ = LineSegment_2D(Point_2D(F, 1, 2), Point_2D(F, 1, 1//10))
            npoints, p₁ = l₁ ∩ l₂
            @test npoints == 0
            @test p₁ ≈ Point_2D(F, 1, 0) # the closest point on line 1

            # line intersects, not segment (invalid s)
            l₂ = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 2, 0    ))
            l₁ = LineSegment_2D(Point_2D(F, 1, 2), Point_2D(F, 1, 1//10))
            npoints, p₁ = intersect(l₁, l₂)
            @test npoints == 0
            @test p₁ ≈ Point_2D(F, 1, 0) # the closest point on line 1

            # is_left
            l = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 1, 0))
            @test is_left(Point_2D(F, 0, 1) , l)
            @test !is_left(Point_2D(F, 0, -1) , l)
            @test !is_left(Point_2D(F, 0, -1e-6) , l)
            @test is_left(Point_2D(F, 0, 1e-6) , l)
        end
    end
end
