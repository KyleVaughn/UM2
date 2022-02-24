        # isleft
        l = LineSegment2D(Point2D{T}(0, 0), Point2D{T}(1, 0))
        @test isleft(Point2D{T}(0, 1) , l)
        @test !isleft(Point2D{T}(0, -1) , l)
        @test !isleft(Point2D{T}(0, -1e-6) , l)
        @test isleft(Point2D{T}(0, 1e-6) , l)
        @test isleft(Point2D{T}(0.5, 0) , l)

        # nearest_point
        𝘅₁ = Point2D{T}(0, 0)           
        𝘅₂ = Point2D{T}(2, 0)
        𝘅₃ = Point2D{T}(1, 1)
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        p = Point2D{T}(1, 1.1)
        r, p_c = nearest_point(p, q)
        @test r ≈ 0.5
        @test 𝘅₃ ≈ p_c

        # isleft
        𝘅₁ = Point2D{T}(0, 0)
        𝘅₂ = Point2D{T}(2, 0)
        𝘅₃ = Point2D{T}(1, 1)
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        @test !isleft(Point2D{T}(1, 0), q)
        @test isleft(Point2D{T}(1, 2), q)
        @test !isleft(Point2D{T}(1, 0.9), q)

        # in
        @test Point2D{F}(0,0) ∈ plane
        @test Point2D{F}(4,4) ∈ plane
        @test Point2D{F}(1,2) ∉ plane

        # in_halfspace
        @test in_halfspace(Point2D{F}(0,0), plane)
        @test in_halfspace(Point2D{F}(0,1), plane)
        @test !in_halfspace(Point2D{F}(0,-1), plane)

        # in 
        @test Point3D{F}(1,0,2) ∈ plane
        @test Point3D{F}(2,2,2) ∈ plane
        @test Point3D{F}(1,0,0) ∉ plane

        # in_halfspace
        @test in_halfspace(Point3D{F}(0,0,2), plane)
        @test in_halfspace(Point3D{F}(0,0,3), plane)
        @test !in_halfspace(Point3D{F}(0,0,-1), plane)
