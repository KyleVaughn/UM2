using MOCNeutronTransport
@testset "AAB2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Methods" begin
            # getproperty
            aab = AAB2D(Point2D{F}(1, 0), Point2D{F}(3, 2))
            @test aab.xmin ≈ 1
            @test aab.ymin ≈ 0
            @test aab.xmax ≈ 3
            @test aab.ymax ≈ 2

            # width, height, area
            @test width(aab) ≈ 2
            @test height(aab) ≈ 2
            @test area(aab) ≈ 4

            # in 
            @test Point2D{F}(2, 1) ∈ aab
            @test Point2D{F}(3, 1) ∈ aab
            @test !(Point2D{F}(4, 1) ∈ aab)
            @test !(Point2D{F}(2, 5) ∈ aab)

            # intersect
            # Horizontal
            hit, points = LineSegment2D(Point2D{F}(-1, 1), Point2D{F}(4, 1)) ∩ aab
            @test hit
            @test points[1] ≈ Point2D{F}(1, 1)
            @test points[2] ≈ Point2D{F}(3, 1)

            # Horizontal miss
            hit, points = LineSegment2D(Point2D{F}(-1, 5), Point2D{F}(4, 5)) ∩ aab
            @test !hit

            # Vertical
            hit, points = LineSegment2D(Point2D{F}(2, -1), Point2D{F}(2, 5)) ∩ aab
            @test hit
            @test points[1] ≈ Point2D{F}(2, 0)
            @test points[2] ≈ Point2D{F}(2, 2)

            # Vertical miss
            hit, points = LineSegment2D(Point2D{F}(5, -1), Point2D{F}(5, 5)) ∩ aab
            @test !hit

            # Angled
            hit, points = LineSegment2D(Point2D{F}(0, 0), Point2D{F}(5, 2)) ∩ aab
            @test hit
            @test points[1] ≈ Point2D{F}(1, 0.4)
            @test points[2] ≈ Point2D{F}(3, 1.2)

            # Angled miss
            hit, points = LineSegment2D(Point2D{F}(0, 5), Point2D{F}(5, 7)) ∩ aab
            @test !hit

            # union
            r1 = AAB2D(Point2D{F}(0, 0), Point2D{F}(2, 2)) 
            r2 = AAB2D(Point2D{F}(1, 1), Point2D{F}(4, 4)) 
            r3 = r1 ∪ r2
            @test r3.xmin ≈ 0
            @test r3.ymin ≈ 0
            @test r3.xmax ≈ 4
            @test r3.ymax ≈ 4
        end
    end
end
