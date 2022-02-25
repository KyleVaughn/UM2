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

        # intersect
        # Horizontal
        hit, points = LineSegment2D(Point2D{T}(-1, 1), Point2D{T}(4, 1)) ∩ aab
        @test hit
        @test points[1] ≈ Point2D{T}(1, 1)
        @test points[2] ≈ Point2D{T}(3, 1)

        # Horizontal miss
        hit, points = LineSegment2D(Point2D{T}(-1, 5), Point2D{T}(4, 5)) ∩ aab
        @test !hit

        # Vertical
        hit, points = LineSegment2D(Point2D{T}(2, -1), Point2D{T}(2, 5)) ∩ aab
        @test hit
        @test points[1] ≈ Point2D{T}(2, 0)
        @test points[2] ≈ Point2D{T}(2, 2)

        # Vertical miss
        hit, points = LineSegment2D(Point2D{T}(5, -1), Point2D{T}(5, 5)) ∩ aab
        @test !hit

        # Angled
        hit, points = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(5, 2)) ∩ aab
        @test hit
        @test points[1] ≈ Point2D{T}(1, 0.4)
        @test points[2] ≈ Point2D{T}(3, 1.2)

        # Angled miss
        hit, points = LineSegment2D(Point2D{T}(0, 5), Point2D{T}(5, 7)) ∩ aab
        @test !hit

        # intersect
        # Horizontal
        hit, points = LineSegment3D(Point3D{T}(-1, 1, 0), Point3D{T}(4, 1, 0)) ∩ aab
        @test hit
        @test points[1] ≈ Point3D{T}(1, 1, 0)
        @test points[2] ≈ Point3D{T}(3, 1, 0)

        # Horizontal miss
        hit, points = LineSegment3D(Point3D{T}(-1, 5, 0), Point3D{T}(4, 5, 0)) ∩ aab
        @test !hit

        # Vertical
        hit, points = LineSegment3D(Point3D{T}(2, -1, 0), Point3D{T}(2, 5, 0)) ∩ aab
        @test hit
        @test points[1] ≈ Point3D{T}(2, 0, 0)
        @test points[2] ≈ Point3D{T}(2, 2, 0)

        # Vertical miss
        hit, points = LineSegment3D(Point3D{T}(5, -1, 0), Point3D{T}(5, 5, 0)) ∩ aab
        @test !hit

        # Angled
        hit, points = LineSegment3D(Point3D{T}(0, 0, -1), Point3D{T}(5, 2, 1)) ∩ aab
        @test hit
        @test points[1] ≈ Point3D{T}(1, 0.4, -0.6)
        @test points[2] ≈ Point3D{T}(3, 1.2,  0.2)

        # Angled miss
        hit, points = LineSegment3D(Point3D{T}(0, 5, -10), Point3D{T}(5, 7, 20)) ∩ aab
        @test !hit


                # intersect
        # 3 intersections
        l = LineSegment2D(p₁, Point2D{T}(1, 1))
        hit, points = intersect(l, tri)
        @test hit
        @test points[1] ≈ p₁
        @test points[2] ≈ Point2D{T}(1//2, 1//2)

        # 2 intersections
        l = LineSegment2D(Point2D{T}(0, 1//2), Point2D{T}(1//2, 0))
        hit, points = intersect(l, tri)
        @test hit
        @test points[1] ≈ Point2D{T}(1//2, 0)
        @test points[2] ≈ Point2D{T}(0, 1//2)

        # 0 intersections
        l = LineSegment2D(Point2D{T}(-1, -1), Point2D{T}(2, -1))
        hit, points = intersect(l, tri)
        @test !hit


                # intersect
        # line is not coplanar with triangle
        p₄ = Point3D{T}(-1, 1//10, 1//10)
        p₅ = Point3D{T}( 1, 1//10, 1//10)
        l = LineSegment(p₄, p₅)
        hit, point = intersect(l, tri)
        @test hit
        @test point ≈ Point3D{T}(0, 1//10,  1//10)

        # line is coplanar with triangle
        p₄ = Point3D{T}(0, -1, 1//10)
        p₅ = Point3D{T}(0,  2, 1//10)
        l = LineSegment(p₄, p₅)
        hit, point = intersect(l, tri)
        @test !hit

        # no intersection non-coplanar
        p₄ = Point3D{T}(-1, 1//10, -1//10)
        p₅ = Point3D{T}( 1, 1//10, -1//10)
        l = LineSegment(p₄, p₅)
        hit, point = intersect(l, tri)
        @test !hit

        # no intersection coplanar
        p₄ = Point3D{T}(0, -1, 1)
        p₅ = Point3D{T}(0, -1, 0)
        l = LineSegment(p₄, p₅)
        hit, point = intersect(l, tri)
        @test !hit

        # intersects on boundary of triangle
        p₄ = Point3D{T}(-1, 0, 0)
        p₅ = Point3D{T}( 1, 0, 0)
        l = LineSegment(p₄, p₅)
        hit, point = intersect(l, tri)
        @test hit
        @test point ≈ Point3D{T}(0, 0, 0)


                # 4 intersections
        l = LineSegment2D(p₃, p₁)
        hit, points = intersect(l, quad)
        @test hit
        @test points[1] ≈ p₁
        @test points[2] ≈ p₃

        # 2 intersections
        l = LineSegment2D(Point2D{T}(0, 1//2), Point2D{T}(1, 1//2))
        hit, points = intersect(l, quad)
        @test hit
        @test points[1] ≈ Point2D{T}(1, 1//2)
        @test points[2] ≈ Point2D{T}(0, 1//2)

        # 0 intersections
        l = LineSegment2D(Point2D{T}(-1, -1), Point2D{T}(2, -1))
        hit, points = intersect(l, quad)
        @test !hit


                    # intersect
            # 0 intersection
            l = LineSegment2D(Point2D{T}(0, -1), Point2D{T}(4, -1))
            n, points = l ∩ tri6
            @test n == 0

            # 2 intersection
            l = LineSegment2D(Point2D{T}(0, 1), Point2D{T}(4, 1))
            n, points = l ∩ tri6
            @test n == 2
            @test points[1] ≈ Point2D{T}(3, 1)
            @test points[2] ≈ Point2D{T}(1, 1)

            # 4 intersection
            l = LineSegment2D(Point2D{T}(0, 1//10), Point2D{T}(4, 1//10))
            n, points = l ∩ tri6
            @test n == 4
            @test points[1] ≈ Point2D{T}(0.4254033307585166, 1//10)
            @test points[2] ≈ Point2D{T}(1.9745966692414834, 1//10)
            @test points[3] ≈ Point2D{T}(2.1900000000000000, 1//10)
            @test points[4] ≈ Point2D{T}(1//10,              1//10)

            # 6 intersection
            p₁ = Point2D{T}( 1, 0)
            p₂ = Point2D{T}( 0, 0)
            p₃ = Point2D{T}(-1, 0)
            p₄ = Point2D{T}( 1//2, -1//2)
            p₅ = Point2D{T}(-1//2, -1//2)
            p₆ = Point2D{T}(  0,    -2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            l = LineSegment2D(Point2D{T}(-2, -1//4), Point2D{T}(2, -1//4))
            n, points = l ∩ tri6
            @test n == 6
            @test points[1] ≈ Point2D{T}( 0.14644659, -1//4)
            @test points[2] ≈ Point2D{T}( 0.8535534,  -1//4)
            @test points[3] ≈ Point2D{T}(-0.8535534,  -1//4)
            @test points[4] ≈ Point2D{T}(-0.14644665, -1//4)
            @test points[5] ≈ Point2D{T}( 0.9354143,  -1//4)
            @test points[6] ≈ Point2D{T}(-0.9354143,  -1//4)

