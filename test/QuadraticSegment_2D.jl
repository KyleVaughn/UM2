using MOCNeutronTransport
@testset "QuadraticSegment_2D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point_2D(T, 0, 0)
            x⃗₂ = Point_2D(T, 2, 0)
            x⃗₃ = Point_2D(T, 1, 1)

            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
        end

        @testset "Methods" begin
            # interpolation
            x⃗₁ = Point_2D(T, 0, 0)
            x⃗₂ = Point_2D(T, 2, 0)
            x⃗₃ = Point_2D(T, 1, 1)

            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
            for t = LinRange{T}(0, 1, 11)
                @test q(t) ≈ Point_2D(T, 2t, -(2t)^2 + 4t)
            end

            # arc_length
            x⃗₁ = Point_2D(T, 0, 0,)
            x⃗₂ = Point_2D(T, 2, 0,)
            x⃗₃ = Point_2D(T, 1, 0,)
            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)

            # straight edge
            @test abs(arc_length(q) - T(2)) < 1.0e-6
            # curved
            x⃗₃ = Point_2D(T, 1, 1)
            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
            @test abs(arc_length(q) - T(2.9578857151786138)) < 1.0e-6

            # intersect
            x⃗₁ = Point_2D(T, 0, 0)
            x⃗₂ = Point_2D(T, 2, 0)
            x⃗₃ = Point_2D(T, 1, 1)
            x⃗₄ = Point_2D(T, 1, 0)
            x⃗₅ = Point_2D(T, 1, 2)

            # 1 intersection
            q = QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃)
            l = LineSegment_2D(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test bool
            @test npoints == 1
            @test points[1] ≈ Point_2D(T, 1, 1)

            # 2 intersections
            x⃗₄ = Point_2D(T, 0, 3//4)
            x⃗₅ = Point_2D(T, 2, 3//4)
            l = LineSegment_2D(x⃗₄, x⃗₅)
            bool, npoints, points = l ∩ q
            @test bool
            @test npoints == 2
            @test points[1] ≈ Point_2D(T, 1//2, 3//4)
            @test points[2] ≈ Point_2D(T, 3//2, 3//4)

            # 0 intersections
            x⃗₄ = Point_2D(T, 0, 3)
            x⃗₅ = Point_2D(T, 2, 3)
            l = LineSegment_2D(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test !bool
            @test npoints == 0
            @test points[1] ≈ Point_2D(T, 1e9, 1e9)
            @test points[2] ≈ Point_2D(T, 1e9, 1e9)
        end
    end
end
