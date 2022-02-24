        # intersect
        # -------------------------------------------
        # basic intersection
        l₁ = LineSegment2D(Point2D{T}(0,  1), Point2D{T}(2, -1))
        l₂ = LineSegment2D(Point2D{T}(0, -1), Point2D{T}(2,  1))
        hit, p₁ = intersect(l₁, l₂)
        @test hit
        @test p₁ ≈ Point2D{T}(1, 0)
        @test typeof(p₁) == Point2D{T}

        # vertex intersection
        l₂ = LineSegment2D(Point2D{T}(0, -1), Point2D{T}(2, -1))
        hit, p₁ = l₁ ∩ l₂
        @test hit
        @test p₁ ≈ Point2D{T}(2, -1)

        # vertical
        l₁ = LineSegment2D(Point2D{T}(0,  1), Point2D{T}(2,   1))
        l₂ = LineSegment2D(Point2D{T}(1, 10), Point2D{T}(1, -10))
        hit, p₁ = intersect(l₁, l₂)
        @test hit
        @test p₁ ≈ Point2D{T}(1, 1)

        # nearly vertical
        l₁ = LineSegment2D(Point2D{T}(-1, -100000), Point2D{T}(1,  100000))
        l₂ = LineSegment2D(Point2D{T}(-1,   10000), Point2D{T}(1,  -10000))
        hit, p₁ = l₁ ∩ l₂
        @test hit
        @test p₁ ≈ Point2D{T}(0, 0)

        # parallel
        l₁ = LineSegment2D(Point2D{T}(0, 1), Point2D{T}(1, 1))
        l₂ = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(1, 0))
        hit, p₁ = intersect(l₁, l₂)
        @test !hit

        # collinear
        l₁ = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(2, 0))
        l₂ = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(1, 0))
        hit, p₁ = intersect(l₁, l₂)
        @test !hit

        # line intersects, not segment (invalid s)
        l₁ = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(2, 0    ))
        l₂ = LineSegment2D(Point2D{T}(1, 2), Point2D{T}(1, 1//10))
        hit, p₁ = l₁ ∩ l₂
        @test !hit

        # line intersects, not segment (invalid r)
        l₂ = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(2, 0    ))
        l₁ = LineSegment2D(Point2D{F}(1, 2), Point2D{F}(1, 1//10))
        hit, p₁ = intersect(l₁, l₂)
        @test !hit

        






        # intersect
        l₁ = LineSegment3D(Point3D{T}(0,  1, 0), Point3D{T}(2, -1, 0))
        l₂ = LineSegment3D(Point3D{T}(0, -1, 0), Point3D{T}(2,  1, 0))
        hit, p₁ = intersect(l₁, l₂)
        @test hit
        @test p₁ ≈ Point3D{T}(1, 0, 0)
        @test typeof(p₁) == Point3D{T}









        # intersect
        𝘅₁ = Point2D{T}(0, 0)
        𝘅₂ = Point2D{T}(2, 0)
        𝘅₃ = Point2D{T}(1, 1)
        𝘅₄ = Point2D{T}(1, 0)
        𝘅₅ = Point2D{T}(1, 2)

        # 1 intersection, straight
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, Point2D{T}(1//2, 0))
        l = LineSegment2D(Point2D{T}(1,-1), Point2D{T}(1,1))
        npoints, (point1, point2) = intersect(l, q)
        @test npoints == 1
        @test point1 ≈ Point2D{T}(1, 0)

        # 1 intersection
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        l = LineSegment2D(𝘅₄, 𝘅₅)
        npoints, (point1, point2) = intersect(l, q)
        @test npoints == 1
        @test point1 ≈ Point2D{T}(1, 1)

        # 2 intersections
        𝘅₄ = Point2D{T}(0, 3//4)
        𝘅₅ = Point2D{T}(2, 3//4)
        l = LineSegment2D(𝘅₄, 𝘅₅)
        npoints, (point1, point2) = l ∩ q
        @test npoints == 2
        @test point1 ≈ Point2D{T}(1//2, 3//4)
        @test point2 ≈ Point2D{T}(3//2, 3//4)

        # 0 intersections
        𝘅₄ = Point2D{T}(0, 3)
        𝘅₅ = Point2D{T}(2, 3)
        l = LineSegment2D(𝘅₄, 𝘅₅)
        npoints, (point1, point2) = intersect(l, q)
        @test npoints == 0
        

        # intersect
        hit, point = LineSegment(Point2D{F}(0, 1), Point2D{F}(1, 0)) ∩ plane
        @test hit
        @test point ≈ Point2D{F}(1//2, 1//2)

        # Line is in the plane
        hit, point = LineSegment(Point2D{F}(0, 0), Point2D{F}(1, 1)) ∩ plane
        @test !hit

        # Segment stops before plane
        hit, point = LineSegment(Point2D{F}(0, 2), Point2D{F}(1, 3//2)) ∩ plane
        @test !hit

        # Plane is before segment
        hit, point = LineSegment(Point2D{F}(1, 0), Point2D{F}(2, -1)) ∩ plane
        @test !hit


        # intersect
        hit, point = LineSegment(Point3D{F}(1, 2, 0), Point3D{F}(1, 2, 5)) ∩ plane
        @test hit
        @test point ≈ Point3D{F}(1,2,2)

        # Line is in the plane
        hit, point = LineSegment(Point3D{F}(0, 0, 2), Point3D{F}(1, 0, 2)) ∩ plane
        @test !hit

        # Segment stops before plane
        hit, point = LineSegment(Point3D{F}(1, 2, 0), Point3D{F}(1, 2, 1)) ∩ plane
        @test !hit

        # Plane is before segment
        hit, point = LineSegment(Point3D{F}(1, 2, 1), Point3D{F}(1, 2, 0)) ∩ plane
        @test !hit

        #isleft
        l = LineSegment(Point3D{F}(0, 0, 2), Point3D{F}(1, 0, 2))
        @test isleft(Point3D{F}(1,1,2), l, plane)
        @test isleft(Point3D{F}(1,0,2), l, plane)
        @test !isleft(Point3D{F}(1,-1,2), l, plane)
