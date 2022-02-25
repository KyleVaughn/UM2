            # interpolation
            p₁ = Point2D{T}(1, 1)
            p₂ = Point2D{T}(3, 3)
            l = LineSegment2D(p₁, p₂) 
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point2D{T}(2, 2)

                        # interpolation
            p₁ = Point3D{T}(1, 1, 1)
            p₂ = Point3D{T}(3, 3, 3)
            l = LineSegment3D(p₁, p₂)
            @test l(0) ≈ p₁
            @test l(1) ≈ p₂
            @test l(1//2) ≈ Point3D{T}(2, 2, 2)

                        # interpolation
            𝘅₁ = Point2D{T}(0, 0)
            𝘅₂ = Point2D{T}(2, 0)
            𝘅₃ = Point2D{T}(1, 1)
            q = QuadraticSegment2D(𝘅₁, 𝘅₂, 𝘅₃)
            for r = LinRange{T}(0, 1, 11)
                @test q(r) ≈ Point2D{T}(2r, -(2r)^2 + 4r)
            end

                        # interpolation
            @test tri(0, 0) ≈ p₁
            @test tri(1, 0) ≈ p₂
            @test tri(0, 1) ≈ p₃
            @test tri(1//2, 1//2) ≈ Point2D{T}(1//2, 1//2)

                        # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point3D{T}(0, 1//2, 1//2)

                        # interpolation
            @test quad(0, 0) ≈ p₁
            @test quad(1, 0) ≈ p₂
            @test quad(1, 1) ≈ p₃
            @test quad(0, 1) ≈ p₄
            @test quad(1//2, 1//2) ≈ Point2D{T}(1//2, 1//2)

                    # interpolation
        @test tri6(0, 0) ≈ p₁
        @test tri6(1, 0) ≈ p₂
        @test tri6(0, 1) ≈ p₃
        @test tri6(1//2, 0) ≈ p₄
        @test tri6(1//2, 1//2) ≈ p₅
        @test tri6(0, 1//2) ≈ p₆

                # interpolation
        @test quad8(0, 0) ≈ p₁
        @test quad8(1, 0) ≈ p₂
        @test quad8(1, 1) ≈ p₃
        @test quad8(0, 1) ≈ p₄
        @test quad8(1//2,    0) ≈ p₅
        @test quad8(   1, 1//2) ≈ p₆
        @test quad8(1//2,    1) ≈ p₇
        @test quad8(   0, 1//2) ≈ p₈
        @test quad8(1//2, 1//2) ≈ Point2D{T}(1//2, 1//2)
