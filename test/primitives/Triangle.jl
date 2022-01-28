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
