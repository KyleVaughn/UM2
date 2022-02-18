using MOCNeutronTransport
@testset "Triangle2D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point2D{F}(0, 0)
            p₂ = Point2D{F}(1, 0)
            p₃ = Point2D{F}(1, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point2D{F}(0, 0)
            p₂ = Point2D{F}(1, 0)
            p₃ = Point2D{F}(0, 1)
            tri = Triangle(p₁, p₂, p₃)

            # interpolation
            @test tri(0, 0) ≈ p₁
            @test tri(1, 0) ≈ p₂
            @test tri(0, 1) ≈ p₃
            @test tri(1//2, 1//2) ≈ Point2D{F}(1//2, 1//2)

            # area
            a = area(tri)
            @test typeof(a) == F
            @test a == F(1//2)

            # centroid
            @test centroid(tri) ≈ Point2D{F}(1//3, 1//3)

            # in
            p = Point2D{F}(1//2, 1//10)
            @test p ∈ tri
            p = Point2D{F}(1//2, -1//10)
            @test p ∉ tri

            # intersect
            # 3 intersections
            l = LineSegment2D(p₁, Point2D{F}(1, 1))
            ipoints, points = intersect(l, tri)
            @test ipoints == 3 
            @test points[1] ≈ p₁
            @test points[2] ≈ Point2D{F}(1//2, 1//2)
            @test points[3] ≈ p₁ 

            # 2 intersections
            l = LineSegment2D(Point2D{F}(0, 1//2), Point2D{F}(1//2, 0))
            ipoints, points = intersect(l, tri)
            @test ipoints == 2 
            @test points[1] ≈ Point2D{F}(1//2, 0)
            @test points[2] ≈ Point2D{F}(0, 1//2)

            # 0 intersections
            l = LineSegment2D(Point2D{F}(-1, -1), Point2D{F}(2, -1))
            ipoints, points = intersect(l, tri)                    
            @test ipoints == 0 
        end
    end
end

@testset "Triangle3D" begin
    for F in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point3D{F}(0, 0, 0)
            p₂ = Point3D{F}(0, 1, 0)
            p₃ = Point3D{F}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point3D{F}(0, 0, 0)
            p₂ = Point3D{F}(0, 1, 0)
            p₃ = Point3D{F}(0, 0, 1)
            tri = Triangle(p₁, p₂, p₃)

            # interpolation
            tri(0, 0) ≈ p₁
            tri(1, 0) ≈ p₂
            tri(0, 1) ≈ p₃
            tri(1//2, 1//2) ≈ Point3D{F}(0, 1//2, 1//2)

            # area
            @test area(tri) ≈ 1//2

            # centroid
            @test centroid(tri) ≈ Point3D{F}(0, 1//3, 1//3)

            # intersect
            # line is not coplanar with triangle
            p₄ = Point3D{F}(-1, 1//10, 1//10)
            p₅ = Point3D{F}( 1, 1//10, 1//10)
            l = LineSegment(p₄, p₅)
            hit, point = intersect(l, tri)
            @test hit
            @test point ≈ Point3D{F}(0, 1//10,  1//10)

            # line is coplanar with triangle
            p₄ = Point3D{F}(0, -1, 1//10)
            p₅ = Point3D{F}(0,  2, 1//10)
            l = LineSegment(p₄, p₅)
            hit, point = intersect(l, tri)
            @test !hit

            # no intersection non-coplanar
            p₄ = Point3D{F}(-1, 1//10, -1//10)
            p₅ = Point3D{F}( 1, 1//10, -1//10)
            l = LineSegment(p₄, p₅)
            hit, point = intersect(l, tri)
            @test !hit

            # no intersection coplanar
            p₄ = Point3D{F}(0, -1, 1)
            p₅ = Point3D{F}(0, -1, 0)
            l = LineSegment(p₄, p₅)
            hit, point = intersect(l, tri)
            @test !hit

            # intersects on boundary of triangle
            p₄ = Point3D{F}(-1, 0, 0)
            p₅ = Point3D{F}( 1, 0, 0)
            l = LineSegment(p₄, p₅)
            hit, point = intersect(l, tri)
            @test hit
            @test point ≈ Point3D{F}(0, 0, 0)
        end
    end
end
