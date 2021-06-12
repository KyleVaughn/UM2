using MOCNeutronTransport
@testset "QuadraticSegment" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point(T, 0, 0, 0)
            x⃗₂ = Point(T, 2, 0, 0)
            x⃗₃ = Point(T, 1, 1, 0)

            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
        end

        @testset "Methods" begin
            # interpolation
            x⃗₁ = Point(T, 0, 0, 0)
            x⃗₂ = Point(T, 2, 0, 0)
            x⃗₃ = Point(T, 1, 1, 0)

            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
            for t = LinRange{T}(0, 1, 11)
                @test q(t) ≈ Point(T, 2t, -(2t)^2 + 4t)
            end

            # arc_length
            x⃗₁ = Point(T, 0, 0, 0)
            x⃗₂ = Point(T, 2, 0, 0)
            x⃗₃ = Point(T, 1, 0, 0)
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)

            # straight edge
            @test abs(arc_length(q) - T(2)) < 1.0e-6
            # curved
            x⃗₃ = Point(T, 1, 1, 0)
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            @test abs(arc_length(q) - T(2.9578857151786138)) < 1.0e-6

            # intersect
            x⃗₁ = Point(T, 0, 0, 0)
            x⃗₂ = Point(T, 2, 0, 0)
            x⃗₃ = Point(T, 1, 1, 0)
            x⃗₄ = Point(T, 1, 0, 0)
            x⃗₅ = Point(T, 1, 2, 0)

            # 1 intersection
            q = QuadraticSegment(x⃗₁, x⃗₂, x⃗₃)
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test bool
            @test npoints == 1
            @test points[1] ≈ Point(T, 1, 1, 0)

            # 2 intersections
            x⃗₄ = Point(T, 0, 3//4, 0)
            x⃗₅ = Point(T, 2, 3//4, 0)
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = l ∩ q
            @test bool
            @test npoints == 2
            @test points[1] ≈ Point(T, 1//2, 3//4, 0)
            @test points[2] ≈ Point(T, 3//2, 3//4, 0)

            # 0 intersections
            x⃗₄ = Point(T, 0, 3, 0)
            x⃗₅ = Point(T, 2, 3, 0)
            l = LineSegment(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test !bool
            @test npoints == 0
            @test points[1] ≈ Point(T, 1e9, 1e9, 1e9)
            @test points[2] ≈ Point(T, 1e9, 1e9, 1e9)
        end
    end
end
