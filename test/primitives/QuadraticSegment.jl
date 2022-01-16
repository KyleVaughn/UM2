using MOCNeutronTransport
@testset "QuadraticSegment_2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            # Constructor
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test q.points == SVector(𝘅₁, 𝘅₂, 𝘅₃)
        end

        @testset "Methods" begin
            # interpolation
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            for r = LinRange{F}(0, 1, 11)
                @test q(r) ≈ Point_2D{F}(2r, -(2r)^2 + 4r)
            end
 
            # derivative
            for r = LinRange{F}(0, 1, 11)
                @test 𝗗(q, r) ≈ SVector{2,F}(2, -(8r) + 4)
            end

            # arclength
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 0)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            # straight edge
            @test abs(arclength(q) - 2) < 1.0e-6
            # curved
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test abs(arclength(q) - 2.9578857151786138) < 1.0e-6
 
            # boundingbox
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            bb = boundingbox(q)
            @test bb.xmin ≈ 0
            @test bb.ymin ≈ 0
            @test bb.xmax ≈ 2
            @test bb.ymax ≈ 1
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 2)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            bb = boundingbox(q)
            @test bb.xmin ≈ 0
            @test bb.ymin ≈ 0
            @test bb.xmax ≈ 2
            @test bb.ymax ≈ 2
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(2.1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            bb = boundingbox(q)
            @test bb.xmin ≈ 0
            @test bb.ymin ≈ 0
            @test bb.xmax ≈ 2.3272727272727276
            @test bb.ymax ≈ 1
 
            # isstraight
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 0)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test isstraight(q)
            𝘅₂ = Point_2D{F}(2, 0.0001)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test !isstraight(q)


            # closest_point
            𝘅₁ = Point_2D{F}(0, 0)           
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            p = Point_2D{F}(1, 1.1)
            r, p_c = nearest_point(p, q)
            @test r ≈ 0.5
            @test 𝘅₃ ≈ p_c

            # isleft
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            @test !isleft(Point_2D{F}(1, 0), q)
            @test isleft(Point_2D{F}(1, 2), q)
            @test !isleft(Point_2D{F}(1, 0.9), q)


            # intersect
            𝘅₁ = Point_2D{F}(0, 0)
            𝘅₂ = Point_2D{F}(2, 0)
            𝘅₃ = Point_2D{F}(1, 1)
            𝘅₄ = Point_2D{F}(1, 0)
            𝘅₅ = Point_2D{F}(1, 2)

            # 1 intersection
            q = QuadraticSegment_2D(𝘅₁, 𝘅₂, 𝘅₃)
            l = LineSegment_2D(𝘅₄, 𝘅₅)
            npoints, (point1, point2) = intersect(l, q)
            @test npoints == 1
            @test point1 ≈ Point_2D{F}(1, 1)

            # 2 intersections
            𝘅₄ = Point_2D{F}(0, 3//4)
            𝘅₅ = Point_2D{F}(2, 3//4)
            l = LineSegment_2D(𝘅₄, 𝘅₅)
            npoints, (point1, point2) = l ∩ q
            @test npoints == 2
            @test point1 ≈ Point_2D{F}(1//2, 3//4)
            @test point2 ≈ Point_2D{F}(3//2, 3//4)

            # 0 intersections
            𝘅₄ = Point_2D{F}(0, 3)
            𝘅₅ = Point_2D{F}(2, 3)
            l = LineSegment_2D(𝘅₄, 𝘅₅)
            npoints, (point1, point2) = intersect(l, q)
            @test npoints == 0
            @test point1 ≈ Point_2D{F}(0)
            @test point2 ≈ Point_2D{F}(0)
        end
    end
end
