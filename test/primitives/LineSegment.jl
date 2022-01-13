using MOCNeutronTransport
@testset "LineSegment_2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_2D{F}(1, 0)
            p₂ = Point_2D{F}(2, 0)
            l = LineSegment_2D(p₁, p₂)
            @test l[1] == p₁
            @test l[2] == p₂
        end
        @testset "Methods" begin
            # interpolation
            p₁ = Point_2D{F}(1, 1)
            p₂ = Point_2D{F}(3, 3)
            l = LineSegment_2D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point_2D{F}(2, 2)

            # arclength
            p₁ = Point_2D{F}(1, 2)
            p₂ = Point_2D{F}(2, 4)
            l = LineSegment_2D(p₁, p₂)
            @test arclength(l) ≈ sqrt(5)
            @test typeof(arclength(l)) == F

            # Point addition
            l_shift = l + p₁
            @test l_shift[1] ≈ Point_2D{F}(2, 4)
            @test l_shift[2] ≈ Point_2D{F}(3, 6)

            # intersect
            # -------------------------------------------
            # basic intersection
            l₁ = LineSegment_2D(Point_2D{F}(0,  1), Point_2D{F}(2, -1))
            l₂ = LineSegment_2D(Point_2D{F}(0, -1), Point_2D{F}(2,  1))
            hit, p₁ = intersect(l₁, l₂)
            @test hit
            @test p₁ == Point_2D{F}(1, 0)
            @test typeof(p₁) == Point_2D{F}

            # vertex intersection
            l₂ = LineSegment_2D(Point_2D{F}(0, -1), Point_2D{F}(2, -1))
            hit, p₁ = l₁ ∩ l₂
            @test hit
            @test p₁ ≈ Point_2D{F}(2, -1)

            # vertical
            l₁ = LineSegment_2D(Point_2D{F}(0,  1), Point_2D{F}(2,   1))
            l₂ = LineSegment_2D(Point_2D{F}(1, 10), Point_2D{F}(1, -10))
            hit, p₁ = intersect(l₁, l₂)
            @test hit
            @test p₁ ≈ Point_2D{F}(1, 1)

            # nearly vertical
            l₁ = LineSegment_2D(Point_2D{F}(-1, -100000), Point_2D{F}(1,  100000))
            l₂ = LineSegment_2D(Point_2D{F}(-1,   10000), Point_2D{F}(1,  -10000))
            hit, p₁ = l₁ ∩ l₂
            @test hit
            @test p₁ ≈ Point_2D{F}(0, 0)

            # parallel
            l₁ = LineSegment_2D(Point_2D{F}(0, 1), Point_2D{F}(1, 1))
            l₂ = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(1, 0))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # collinear
            l₁ = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(2, 0))
            l₂ = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(1, 0))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # line intersects, not segment (invalid s)
            l₁ = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(2, 0    ))
            l₂ = LineSegment_2D(Point_2D{F}(1, 2), Point_2D{F}(1, 1//10))
            hit, p₁ = l₁ ∩ l₂
            @test !hit

            # line intersects, not segment (invalid r)
            l₂ = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(2, 0    ))
            l₁ = LineSegment_2D(Point_2D{F}(1, 2), Point_2D{F}(1, 1//10))
            hit, p₁ = intersect(l₁, l₂)
            @test !hit

            # isleft
            l = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(1, 0))
            @test isleft(Point_2D{F}(0, 1) , l)
            @test !isleft(Point_2D{F}(0, -1) , l)
            @test !isleft(Point_2D{F}(0, -1e-6) , l)
            @test isleft(Point_2D{F}(0, 1e-6) , l)
            @test isleft(Point_2D{F}(0.5, 0) , l)
        end
    end
end
