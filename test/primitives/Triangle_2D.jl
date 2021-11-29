using MOCNeutronTransport
@testset "Triangle_2D" begin
    for F in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            tri = Triangle_2D(SVector(p₁, p₂, p₃))
            @test tri.points == SVector(p₁, p₂, p₃)

            # single constructor
            tri = Triangle_2D(p₁, p₂, p₃)
            @test tri.points == SVector(p₁, p₂, p₃)
        end

        @testset "Methods" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            tri = Triangle_2D(SVector(p₁, p₂, p₃))

            # interpolation
            @test tri(0, 0) ≈ p₁
            @test tri(1, 0) ≈ p₂
            @test tri(0, 1) ≈ p₃
            @test tri(1//2, 1//2) ≈ Point_2D(F, 1, 1//2)

            # area
            a = area(tri)
            @test typeof(a) == F
            @test a == F(1//2)

            # in
            p = Point_2D(F, 1//2, 1//10)
            @test p ∈  tri
            p = Point_2D(F, 1//2, 0)
            @test p ∈  tri
            p = Point_2D(F, 1//2, -1//10)
            @test p ∉ tri

            # intersect
            # 2 intersections
            l = LineSegment_2D(Point_2D(F, 2, 1), p₁)
            ipoints, points = intersect(l, tri)
            @test ipoints == 2 
            @test points[1] ≈ p₁
            @test points[2] ≈ Point_2D(F, 1, 1//2)

            # 1 intersections
            l = LineSegment_2D(Point_2D(F, -1, -1), p₁)
            ipoints, points = intersect(l, tri)
            @test ipoints == 1 
            @test points[1] ≈ p₁

            # 0 intersections
            l = LineSegment_2D(Point_2D(F, -1, -1), Point_2D(F, 2, -1))
            ipoints, points = intersect(l, tri)                    
            @test ipoints == 0 
        end
    end
end