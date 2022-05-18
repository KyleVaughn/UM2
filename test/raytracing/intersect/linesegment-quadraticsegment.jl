 @testset "LineSegment-QuadraticSegment" begin              
     for T ∈ Floats
        # intersect
        X₁ = Point{2,T}(0, 0)
        X₂ = Point{2,T}(2, 0)
        X₃ = Point{2,T}(1, 1)
        X₄ = Point{2,T}(1, 0)
        X₅ = Point{2,T}(1, 2)

        # 1 intersection, straight
        q = QuadraticSegment(X₁, X₂, Point{2,T}(1//2, 0))
        l = LineSegment(Point{2,T}(1,-1), Point{2,T}(1,1))
        (point1, point2) = intersect(l, q)
        @test point1 ≈ Point{2,T}(1, 0)
        @test point2 ≈ Point{2,T}(1e6, 1e6)

        # 1 intersection
        q = QuadraticSegment(X₁, X₂, X₃)
        l = LineSegment(X₄, X₅)
        (point1, point2) = intersect(l, q)
        @test point1 ≈ Point{2,T}(1, 1)
        @test point2 ≈ Point{2,T}(1e6, 1e6)

        # 2 intersections
        X₄ = Point{2,T}(0, 3//4)
        X₅ = Point{2,T}(2, 3//4)
        l = LineSegment(X₄, X₅)
        (point1, point2) = l ∩ q
        @test point1 ≈ Point{2,T}(1//2, 3//4)
        @test point2 ≈ Point{2,T}(3//2, 3//4)

        # 0 intersections
        X₄ = Point{2,T}(0, 3)
        X₅ = Point{2,T}(2, 3)
        l = LineSegment(X₄, X₅)
        (point1, point2) = intersect(l, q)
        @test point1 ≈ Point{2,T}(1e6, 1e6)
        @test point2 ≈ Point{2,T}(1e6, 1e6)
    end
end
