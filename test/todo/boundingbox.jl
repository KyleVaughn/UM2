        # boundingbox
        𝘅₁ = Point2D{T}(0, 0)
        𝘅₂ = Point2D{T}(2, 0)
        𝘅₃ = Point2D{T}(1, 1)
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        bb = boundingbox(q)
        @test bb.xmin ≈ 0
        @test bb.ymin ≈ 0
        @test bb.xmax ≈ 2
        @test bb.ymax ≈ 1
        𝘅₁ = Point2D{T}(0, 0)
        𝘅₂ = Point2D{T}(2, 2)
        𝘅₃ = Point2D{T}(1, 1)
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        bb = boundingbox(q)
        @test bb.xmin ≈ 0
        @test bb.ymin ≈ 0
        @test bb.xmax ≈ 2
        @test bb.ymax ≈ 2
        𝘅₁ = Point2D{T}(0, 0)
        𝘅₂ = Point2D{T}(2, 0)
        𝘅₃ = Point2D{T}(2.1, 1)
        q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
        bb = boundingbox(q)
        @test bb.xmin ≈ 0
        @test bb.ymin ≈ 0
        @test bb.xmax ≈ 2.3272727272727276
        @test bb.ymax ≈ 1

            # union
            aab = union(AABox2D(Point2D{T}(0,0), Point2D{T}(2, 2)),
                        AABox2D(Point2D{T}(1,1), Point2D{T}(3, 3)))
            @test aab ≈ AABox2D(Point2D{T}(0,0), Point2D{T}(3, 3))
