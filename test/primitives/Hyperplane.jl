using MOCNeutronTransport
@testset "Hyperplane2D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            plane = Hyperplane(Point2D{F}(1,1), Point2D{F}(2,2))
            @test plane.𝗻 ≈ [-sqrt(F(2))/2, sqrt(F(2))/2]
            @test plane.d ≈ 0
        end
        @testset "Methods" begin
            plane = Hyperplane(Point2D{F}(1,1), Point2D{F}(2,2))

            # in
            @test Point2D{F}(0,0) ∈ plane
            @test Point2D{F}(4,4) ∈ plane
            @test Point2D{F}(1,2) ∉ plane

            # in_halfspace
            @test in_halfspace(Point2D{F}(0,0), plane)
            @test in_halfspace(Point2D{F}(0,1), plane)
            @test !in_halfspace(Point2D{F}(0,-1), plane)

            # intersect
            hit, point = LineSegment(Point2D{F}(0, 1), Point2D{F}(1, 0)) ∩ plane
            @test hit
            @test point ≈ Point2D{F}(1//2, 1//2)

            # Line is in the plane
            hit, point = LineSegment(Point2D{F}(0, 0), Point2D{F}(1, 1)) ∩ plane
            @test !hit

            # Segment stops before plane
            hit, point = LineSegment(Point2D{F}(0, 2), Point2D{F}(1, 3//2)) ∩ plane
            @test !hit

            # Plane is before segment
            hit, point = LineSegment(Point2D{F}(1, 0), Point2D{F}(2, -1)) ∩ plane
            @test !hit
        end
    end
end
@testset "Hyperplane3D" begin
    @testset "$F" for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            plane = Hyperplane(Point3D{F}(0,0,2), Point3D{F}(1,0,2), Point3D{F}(0,1,2))
            @test plane.𝗻 ≈ [0,0,1]
            @test plane.d ≈ 2
        end
        @testset "Methods" begin
            plane = Hyperplane(Point3D{F}(0,0,2), Point3D{F}(1,0,2), Point3D{F}(0,1,2))

            # in 
            @test Point3D{F}(1,0,2) ∈ plane
            @test Point3D{F}(2,2,2) ∈ plane
            @test Point3D{F}(1,0,0) ∉ plane

            # in_halfspace
            @test in_halfspace(Point3D{F}(0,0,2), plane)
            @test in_halfspace(Point3D{F}(0,0,3), plane)
            @test !in_halfspace(Point3D{F}(0,0,-1), plane)

            # intersect
            hit, point = LineSegment(Point3D{F}(1, 2, 0), Point3D{F}(1, 2, 5)) ∩ plane
            @test hit
            @test point ≈ Point3D{F}(1,2,2)

            # Line is in the plane
            hit, point = LineSegment(Point3D{F}(0, 0, 2), Point3D{F}(1, 0, 2)) ∩ plane
            @test !hit

            # Segment stops before plane
            hit, point = LineSegment(Point3D{F}(1, 2, 0), Point3D{F}(1, 2, 1)) ∩ plane
            @test !hit

            # Plane is before segment
            hit, point = LineSegment(Point3D{F}(1, 2, 1), Point3D{F}(1, 2, 0)) ∩ plane
            @test !hit

            #isleft
            l = LineSegment(Point3D{F}(0, 0, 2), Point3D{F}(1, 0, 2))
            @test isleft(Point3D{F}(1,1,2), l, plane)
            @test isleft(Point3D{F}(1,0,2), l, plane)
            @test !isleft(Point3D{F}(1,-1,2), l, plane)
        end
    end
end
