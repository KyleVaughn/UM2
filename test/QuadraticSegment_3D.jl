using MOCNeutronTransport
@testset "QuadraticSegment_3D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            x⃗₁ = Point_3D(T, 0, 0, 0)
            x⃗₂ = Point_3D(T, 2, 0, 0)
            x⃗₃ = Point_3D(T, 1, 1, 0)

            q = QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
        end

        @testset "Methods" begin
            # interpolation
            x⃗₁ = Point_3D(T, 0, 0, 0)
            x⃗₂ = Point_3D(T, 2, 0, 0)
            x⃗₃ = Point_3D(T, 1, 1, 0)

            q = QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃)
            @test q.points == (x⃗₁, x⃗₂, x⃗₃)
            for t = LinRange{T}(0, 1, 11)
                @test q(t) ≈ Point_3D(T, 2t, -(2t)^2 + 4t)
            end

            # arc_length
            x⃗₁ = Point_3D(T, 0, 0, 0)
            x⃗₂ = Point_3D(T, 2, 0, 0)
            x⃗₃ = Point_3D(T, 1, 0, 0)
            q = QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃)

            # straight edge
            @test abs(arc_length(q) - T(2)) < 1.0e-6
            # curved
            x⃗₃ = Point_3D(T, 1, 1, 0)
            q = QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃)
            @test abs(arc_length(q) - T(2.9578857151786138)) < 1.0e-6

            # intersect
            x⃗₁ = Point_3D(T, 0, 0, 0)
            x⃗₂ = Point_3D(T, 2, 0, 0)
            x⃗₃ = Point_3D(T, 1, 1, 0)
            x⃗₄ = Point_3D(T, 1, 0, 0)
            x⃗₅ = Point_3D(T, 1, 2, 0)

            # 1 intersection
            q = QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃)
            l = LineSegment_3D(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test bool
            @test npoints == 1
            @test points[1] ≈ Point_3D(T, 1, 1, 0)

            # 2 intersections
            x⃗₄ = Point_3D(T, 0, 3//4, 0)
            x⃗₅ = Point_3D(T, 2, 3//4, 0)
            l = LineSegment_3D(x⃗₄, x⃗₅)
            bool, npoints, points = l ∩ q
            @test bool
            @test npoints == 2
            @test points[1] ≈ Point_3D(T, 1//2, 3//4, 0)
            @test points[2] ≈ Point_3D(T, 3//2, 3//4, 0)

            # 0 intersections
            x⃗₄ = Point_3D(T, 0, 3, 0)
            x⃗₅ = Point_3D(T, 2, 3, 0)
            l = LineSegment_3D(x⃗₄, x⃗₅)
            bool, npoints, points = intersect(l, q)
            @test !bool
            @test npoints == 0
            @test points[1] ≈ Point_3D(T, 1e9, 1e9, 1e9)
            @test points[2] ≈ Point_3D(T, 1e9, 1e9, 1e9)
        end
    end
end
