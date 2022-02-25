        # arclength
        p₁ = Point2D{T}(1, 2)
        p₂ = Point2D{T}(2, 4)
        l = LineSegment2D(p₁, p₂)
        @test arclength(l) ≈ sqrt(5)
        @test typeof(arclength(l)) == T

        # arclength
        p₁ = Point3D{T}(1, 2, 3)
        p₂ = Point3D{T}(2, 4, 6)
        l = LineSegment3D(p₁, p₂)
        @test arclength(l) ≈ sqrt(14)

            # arclength
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 0)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            # straight edge
            @test abs(arclength(q) - 2) < 1.0e-6
            # curved
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test abs(arclength(q) - 2.957885715089195) < 1.0e-6


        @test measure(aab) ≈ 24

                a = area(tri)
        @test typeof(a) == T

                # area
        @test area(tri) ≈ 1//2

                # area
        a = area(quad)
        @test typeof(a) == T
        @test a ≈ T(1)

            # area
            p₀ = Point2D{T}(1,1)
            p₁ = Point2D(Point2D{T}(0, 0) + p₀) 
            p₂ = Point2D(Point2D{T}(2, 0) + p₀) 
            p₃ = Point2D(Point2D{T}(2, 2) + p₀) 
            p₄ = Point2D(Point2D{T}(3//2, 1//4) + p₀) 
            p₅ = Point2D(Point2D{T}(3, 1) + p₀) 
            p₆ = Point2D(Point2D{T}(1, 1) + p₀) 
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆) 
            @test isapprox(area(tri6), 3, atol=1.0e-6)

