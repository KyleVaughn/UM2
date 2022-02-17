using MOCNeutronTransport
@testset "AABox2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Methods" begin
            # getproperty
            aab = AABox2D(Point2D{F}(1, 0), Point2D{F}(3, 2))
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
            r1 = AABox2D(Point2D{F}(0, 0), Point2D{F}(2, 2)) 
            r2 = AABox2D(Point2D{F}(1, 1), Point2D{F}(4, 4)) 
            r3 = r1 ∪ r2
            @test r3.xmin ≈ 0
            @test r3.ymin ≈ 0
            @test r3.xmax ≈ 4
            @test r3.ymax ≈ 4
            r1 = AABox2D(Point2D{F}(0, 0), Point2D{F}(2, 2)) 
            r2 = AABox2D(Point2D{F}(1, 1), Point2D{F}(3, 3)) 
            r3 = AABox2D(Point2D{F}(2, 2), Point2D{F}(4, 4)) 
            r4 = union([r1,r2,r3])
            @test r4.xmin ≈ 0
            @test r4.ymin ≈ 0
            @test r4.xmax ≈ 4
            @test r4.ymax ≈ 4

        end
    end
end
@testset "AABox3D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Methods" begin
            # getproperty
            aab = AABox3D(Point3D{F}(1, 0, -1), Point3D{F}(3, 2, 1))
            @test aab.xmin ≈ 1
            @test aab.ymin ≈ 0
            @test aab.zmin ≈ -1
            @test aab.xmax ≈ 3
            @test aab.ymax ≈ 2
            @test aab.zmax ≈ 1

            # width, height, area
            @test width(aab) ≈ 2
            @test height(aab) ≈ 2
            @test depth(aab) ≈ 2
            @test area(aab) ≈ 24

            # in 
            @test Point3D{F}(2, 1, 0) ∈ aab
            @test Point3D{F}(3, 1, 0) ∈ aab
            @test !(Point3D{F}(4, 1, 0) ∈ aab)
            @test !(Point3D{F}(2, 5, 0) ∈ aab)
            @test !(Point3D{F}(2, 1, 2) ∈ aab)

            # intersect
            # Horizontal
            hit, points = LineSegment3D(Point3D{F}(-1, 1, 0), Point3D{F}(4, 1, 0)) ∩ aab
            @test hit
            @test points[1] ≈ Point3D{F}(1, 1, 0)
            @test points[2] ≈ Point3D{F}(3, 1, 0)

            # Horizontal miss
            hit, points = LineSegment3D(Point3D{F}(-1, 5, 0), Point3D{F}(4, 5, 0)) ∩ aab
            @test !hit

            # Vertical
            hit, points = LineSegment3D(Point3D{F}(2, -1, 0), Point3D{F}(2, 5, 0)) ∩ aab
            @test hit
            @test points[1] ≈ Point3D{F}(2, 0, 0)
            @test points[2] ≈ Point3D{F}(2, 2, 0)

            # Vertical miss
            hit, points = LineSegment3D(Point3D{F}(5, -1, 0), Point3D{F}(5, 5, 0)) ∩ aab
            @test !hit

            # Angled
            hit, points = LineSegment3D(Point3D{F}(0, 0, -1), Point3D{F}(5, 2, 1)) ∩ aab
            @test hit
            @test points[1] ≈ Point3D{F}(1, 0.4, -0.6)
            @test points[2] ≈ Point3D{F}(3, 1.2,  0.2)

            # Angled miss
            hit, points = LineSegment3D(Point3D{F}(0, 5, -10), Point3D{F}(5, 7, 20)) ∩ aab
            @test !hit
#
#            # union
#            r1 = AABox3D(Point3D{F}(0, 0), Point3D{F}(2, 2)) 
#            r2 = AABox3D(Point3D{F}(1, 1), Point3D{F}(4, 4)) 
#            r3 = r1 ∪ r2
#            @test r3.xmin ≈ 0
#            @test r3.ymin ≈ 0
#            @test r3.xmax ≈ 4
#            @test r3.ymax ≈ 4
        end
    end
end
