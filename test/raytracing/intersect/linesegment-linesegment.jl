@testset "LineSegment-LineSegment" begin       
    for T ∈ Floats
        # basic intersection
        l₁ = LineSegment(Point{2,T}(0,  1), Point{2,T}(2, -1))
        l₂ = LineSegment(Point{2,T}(0, -1), Point{2,T}(2,  1))
        p = intersect(l₁, l₂)
        @test p ≈ Point{2,T}(1, 0)
        @test typeof(p) == Point{2,T}

        # vertex intersection
        l₂ = LineSegment(Point{2,T}(0, -1), Point{2,T}(2, -1))
        p = l₁ ∩ l₂
        @test p ≈ Point{2,T}(2, -1)

        # vertical
        l₁ = LineSegment(Point{2,T}(0,  1), Point{2,T}(2,   1))
        l₂ = LineSegment(Point{2,T}(1, 10), Point{2,T}(1, -10))
        p = intersect(l₁, l₂)
        @test p ≈ Point{2,T}(1, 1)

        # nearly vertical
        l₁ = LineSegment(Point{2,T}(-1, -100000), Point{2,T}(1,  100000))
        l₂ = LineSegment(Point{2,T}(-1,   10000), Point{2,T}(1,  -10000))
        p = l₁ ∩ l₂
        @test p ≈ Point{2,T}(0, 0)

        # parallel
        l₁ = LineSegment(Point{2,T}(0, 1), Point{2,T}(1, 1))
        l₂ = LineSegment(Point{2,T}(0, 0), Point{2,T}(1, 0))
        p = intersect(l₁, l₂)
        @test p ≈ Point{2,T}(1e6,1e6)

        # collinear
        l₁ = LineSegment(Point{2,T}(0, 0), Point{2,T}(2, 0))
        l₂ = LineSegment(Point{2,T}(0, 0), Point{2,T}(1, 0))
        p = intersect(l₁, l₂)
        @test p ≈ Point{2,T}(1e6,1e6)

        # line intersects, not segment (invalid s)
        l₁ = LineSegment(Point{2,T}(0, 0), Point{2,T}(2, 0    ))
        l₂ = LineSegment(Point{2,T}(1, 2), Point{2,T}(1, 1//10))
        p = l₁ ∩ l₂
        @test p ≈ Point{2,T}(1e6,1e6)

        # line intersects, not segment (invalid r)
        l₂ = LineSegment(Point{2,T}(0, 0), Point{2,T}(2, 0    ))
        l₁ = LineSegment(Point{2,T}(1, 2), Point{2,T}(1, 1//10))
        p = intersect(l₁, l₂)
        @test p ≈ Point{2,T}(1e6,1e6)
    end
end
