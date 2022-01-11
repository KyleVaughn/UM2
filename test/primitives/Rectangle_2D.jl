using MOCNeutronTransport
@testset "Rectangle_2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Methods" begin
            # getproperty
            rect = Rectangle_2D(Point_2D{F}(1, 0), Point_2D{F}(3, 2))
            @test rect.xmin ≈ 1
            @test rect.ymin ≈ 0
            @test rect.xmax ≈ 3
            @test rect.ymax ≈ 2

            # width, height, area
            @test width(rect) ≈ 2
            @test height(rect) ≈ 2
            @test area(rect) ≈ 4

            # in 
            @test Point_2D{F}(2, 1) ∈ rect
            @test Point_2D{F}(3, 1) ∈ rect
            @test !(Point_2D{F}(4, 1) ∈ rect)
            @test !(Point_2D{F}(2, 5) ∈ rect)

            # intersect
            # Horizontal
            hit, points = LineSegment_2D(Point_2D{F}(-1, 1), Point_2D{F}(4, 1)) ∩ rect
            @test hit
            @test points[1] ≈ Point_2D{F}(1, 1)
            @test points[2] ≈ Point_2D{F}(3, 1)

            # Horizontal miss
            hit, points = LineSegment_2D(Point_2D{F}(-1, 5), Point_2D{F}(4, 5)) ∩ rect
            @test !hit

            # Vertical
            hit, points = LineSegment_2D(Point_2D{F}(2, -1), Point_2D{F}(2, 5)) ∩ rect
            @test hit
            @test points[1] ≈ Point_2D{F}(2, 0)
            @test points[2] ≈ Point_2D{F}(2, 2)

            # Vertical miss
            hit, points = LineSegment_2D(Point_2D{F}(5, -1), Point_2D{F}(5, 5)) ∩ rect
            @test !hit

            # Angled
            hit, points = LineSegment_2D(Point_2D{F}(0, 0), Point_2D{F}(5, 2)) ∩ rect
            @test hit
            @test points[1] ≈ Point_2D{F}(1, 0.4)
            @test points[2] ≈ Point_2D{F}(3, 1.2)

            # Angled miss
            hit, points = LineSegment_2D(Point_2D{F}(0, 5), Point_2D{F}(5, 7)) ∩ rect
            @test !hit

            # union
            r1 = Rectangle_2D(Point_2D{F}(0, 0), Point_2D{F}(2, 2)) 
            r2 = Rectangle_2D(Point_2D{F}(1, 1), Point_2D{F}(4, 4)) 
            r3 = r1 ∪ r2
            @test r3.xmin ≈ 0
            @test r3.ymin ≈ 0
            @test r3.xmax ≈ 4
            @test r3.ymax ≈ 4
        end
    end
end
