using MOCNeutronTransport
@testset "LineSegment2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point2D{F}(1, 0)
            p₂ = Point2D{F}(2, 0)
            l = LineSegment2D(p₁, p₂)
            @test l.𝘅₁== p₁
            @test l.𝘂 == p₂ - p₁
        end
        @testset "Methods" begin
            # interpolation
            p₁ = Point2D{F}(1, 1)
            p₂ = Point2D{F}(3, 3)
            l = LineSegment2D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point2D{F}(2, 2)

            # arclength
            p₁ = Point2D{F}(1, 2)
            p₂ = Point2D{F}(2, 4)
            l = LineSegment2D(p₁, p₂)
            @test arclength(l) ≈ sqrt(5)
            @test typeof(arclength(l)) == F

            # intersect
            # -------------------------------------------
            # basic intersection
            l₁ = LineSegment2D(Point2D{F}(0,  1), Point2D{F}(2, -1))
            l₂ = LineSegment2D(Point2D{F}(0, -1), Point2D{F}(2,  1))
            hit, p₁ = intersect(l₁, l₂)
            @test hit
            @test p₁ ≈ Point2D{F}(1, 0)
            @test typeof(p₁) == Point2D{F}

            # vertex intersection
            l₂ = LineSegment2D(Point2D{F}(0, -1), Point2D{F}(2, -1))
            hit, p₁ = l₁ ∩ l₂
            @test hit
            @test p₁ ≈ Point2D{F}(2, -1)

            # vertical
            l₁ = LineSegment2D(Point2D{F}(0,  1), Point2D{F}(2,   1))
            l₂ = LineSegment2D(Point2D{F}(1, 10), Point2D{F}(1, -10))
            hit, p₁ = intersect(l₁, l₂)
            @test hit
            @test p₁ ≈ Point2D{F}(1, 1)

            # nearly vertical
            l₁ = LineSegment2D(Point2D{F}(-1, -100000), Point2D{F}(1,  100000))
            l₂ = LineSegment2D(Point2D{F}(-1,   10000), Point2D{F}(1,  -10000))
            hit, p₁ = l₁ ∩ l₂
            @test hit
            @test p₁ ≈ Point2D{F}(0, 0)

            # parallel
            l₁ = LineSegment2D(Point2D{F}(0, 1), Point2D{F}(1, 1))
            l₂ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(1, 0))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # collinear
            l₁ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(2, 0))
            l₂ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(1, 0))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # line intersects, not segment (invalid s)
            l₁ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(2, 0    ))
            l₂ = LineSegment2D(Point2D{F}(1, 2), Point2D{F}(1, 1//10))
            hit, p₁ = l₁ ∩ l₂
            @test !hit

            # line intersects, not segment (invalid r)
            l₂ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(2, 0    ))
            l₁ = LineSegment2D(Point2D{F}(1, 2), Point2D{F}(1, 1//10))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # isleft
            l = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(1, 0))
            @test isleft(Point2D{F}(0, 1) , l)
            @test !isleft(Point2D{F}(0, -1) , l)
            @test !isleft(Point2D{F}(0, -1e-6) , l)
            @test isleft(Point2D{F}(0, 1e-6) , l)
            @test isleft(Point2D{F}(0.5, 0) , l)

#            # sortpoints
#            l = LineSegment2D(Point2D{F}(0,0), Point2D{F}(10,0))
#            p₁ = Point2D{F}(1, 0)
#            p₂ = Point2D{F}(2, 0)
#            p₃ = Point2D{F}(3, 0)
#            points = [p₃, p₁, p₂]
#            sortpoints!(l, points)
#            @test points[1] == p₁
#            @test points[2] == p₂
#            @test points[3] == p₃
#
#            # sort_intersection_points
#            l = LineSegment2D(Point2D{F}(0,0), Point2D{F}(10,0))
#            p₁ = Point2D{F}(1, 0)
#            p₂ = Point2D{F}(2, 0)
#            p₃ = Point2D{F}(3, 0)
#            points = [p₃, p₁, p₂, Point2D{F}(1 + 1//1000000, 0)]
#            sort_intersection_points!(l, points)
#            @test points[1] == p₁
#            @test points[2] == p₂
#            @test points[3] == p₃
        end
    end
end
