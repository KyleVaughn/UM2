       # centroid
        @test centroid(tri) ≈ Point2D{T}(1//3, 1//3)

                # centroid
        @test centroid(tri) ≈ Point3D{T}(0, 1//3, 1//3)


   # centroid
        @test centroid(quad) ≈ Point2D{T}(1//2, 1//2)

                    # centroid
            p₁ = Point2D{T}(0, 0)
            p₂ = Point2D{T}(1, 0)
            p₃ = Point2D{T}(0, 1)
            p₄ = Point2D{T}(1//2, 0)
            p₅ = Point2D{T}(1//2, 1//2)
            p₆ = Point2D{T}(0, 1//2)
            tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
            @test centroid(tri6) ≈ tri6(1//3, 1//3)

