        # sort!
        p  = Point(T(0))
        p₁ = Point(T(1))
        p₂ = Point(T(2))
        p₃ = Point(T(3))
        points = [p₃, p₁, p₂]
        sort!(p, points)
        @test points[1] == p₁
        @test points[2] == p₂
        @test points[3] == p₃

        # sort
        p  = Point(T(0))
        p₁ = Point(T(1))
        p₂ = Point(T(2))
        p₃ = Point(T(3))
        points = [p₃, p₁, p₂]
        points_sorted = sort(p, points)
        @test points_sorted[1] == p₁
        @test points_sorted[2] == p₂
        @test points_sorted[3] == p₃

        # sort!
        p  = Point2D{T}(0, 0)
        p₁ = Point2D{T}(1, 0)
        p₂ = Point2D{T}(2, 0)
        p₃ = Point2D{T}(3, 0)
        points = [p₃, p₁, p₂]
        sort!(p, points)
        @test points[1] == p₁
        @test points[2] == p₂
        @test points[3] == p₃

        # sort
        p  = Point2D{T}(0, 0)
        p₁ = Point2D{T}(1, 0)
        p₂ = Point2D{T}(2, 0)
        p₃ = Point2D{T}(3, 0)
        points = [p₃, p₁, p₂]
        points_sorted = sort(p, points)
        @test points_sorted[1] == p₁
        @test points_sorted[2] == p₂
        @test points_sorted[3] == p₃

        # sort!
        p  = Point3D{T}(0, 0, 0)
        p₁ = Point3D{T}(1, 0, 0)
        p₂ = Point3D{T}(2, 0, 0)
        p₃ = Point3D{T}(3, 0, 0)
        points = [p₃, p₁, p₂]
        sort!(p, points)
        @test points[1] == p₁
        @test points[2] == p₂
        @test points[3] == p₃

        # sort
        p  = Point3D{T}(0, 0, 0)
        p₁ = Point3D{T}(1, 0, 0)
        p₂ = Point3D{T}(2, 0, 0)
        p₃ = Point3D{T}(3, 0, 0)
        points = [p₃, p₁, p₂]
        points_sorted = sort(p, points)
        @test points_sorted[1] == p₁
        @test points_sorted[2] == p₂
        @test points_sorted[3] == p₃
