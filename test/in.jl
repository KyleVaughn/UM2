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


        # in 
        @test Point2D{T}(2, 1) ∈ aab
        @test Point2D{T}(3, 1) ∈ aab
        @test !(Point2D{T}(4, 1) ∈ aab)
        @test !(Point2D{T}(2, 5) ∈ aab)


        # in 
        @test Point3D{T}(2, 1, 0) ∈ aab
        @test Point3D{T}(3, 1, 0) ∈ aab
        @test !(Point3D{T}(4, 1, 0) ∈ aab)
        @test !(Point3D{T}(2, 5, 0) ∈ aab)
        @test !(Point3D{T}(2, 1, 2) ∈ aab)


        # in
        p = Point2D{T}(1//2, 1//10)
        @test p ∈ tri
        p = Point2D{T}(1//2, -1//10)
        @test p ∉ tri
        # in
        p = Point3D{T}(0, 1//2, 1//10)
        @test p ∈ tri
        p = Point3D{T}(0, 1//2, -1//10)
        @test p ∉ tri
        p = Point3D{T}(1//100, 1//2, 1//10)
        @test p ∉ tri


        # in
        p = Point2D{T}(1//2, 1//10)
        @test p ∈  quad
        p = Point2D{T}(1//2, -1//10)
        @test p ∉ quad

                    # in
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(2, 0)
            p₃ = Point2D{T}(2, 2)
            p₄ = Point2D{T}(3//2, 1//4)
            p₅ = Point2D{T}(3, 1)
            p₆ = Point2D{T}(1, 1)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            @test Point2D{T}(1, 1//2) ∈  tri6
            @test Point2D{T}(1, 0) ∉  tri6


        # in
        p₁ = Point2D{T}(0, 0)
        p₂ = Point2D{T}(2, 0)
        p₃ = Point2D{T}(2, 3)
        p₄ = Point2D{T}(0, 3)
        p₅ = Point2D{T}(3//2, 1//2)
        p₆ = Point2D{T}(5//2, 3//2)
        p₇ = Point2D{T}(3//2, 5//2)
        p₈ = Point2D{T}(0, 1)
        quad8 = QuadraticQuadrilateral(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈) 
        @test Point2D{T}(1, 1) ∈  quad8
        @test Point2D{T}(1, 0) ∉  quad8
