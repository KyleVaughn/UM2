using MOCNeutronTransport
@testset "Triangle6_3D" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 1//2)
            p₅ = Point_3D(T, 1, 1//2)
            p₆ = Point_3D(T, 1//2, 1//2)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)

            # single constructor
            tri6 = Triangle6_3D(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)
        end

        @testset "Methods" begin
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 1)
            p₃ = Point_3D(T, 1, 1)
            p₄ = Point_3D(T, 1//2)
            p₅ = Point_3D(T, 1, 1//2)
            p₆ = Point_3D(T, 1//2, 1//2)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))

            # interpolation
            tri6(0, 0) ≈ p₁
            tri6(1, 0) ≈ p₂
            tri6(0, 1) ≈ p₃
            tri6(1//2, 0) ≈ p₄
            tri6(1//2, 1//2) ≈ p₅
            tri6(0, 1//2) ≈ p₆
            tri6(1//2, 1//2) ≈ Point_3D(T, 1//2, 1//2)

            # area
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 2)
            p₃ = Point_3D(T, 2, 2)
            p₄ = Point_3D(T, 1, 1//4)
            p₅ = Point_3D(T, 3, 1)
            p₆ = Point_3D(T, 1, 1)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))

            # 3D default
            p₂ = Point_3D(T, 2, 0, 3)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test isapprox(area(tri6; N = 79), 6.328781460309, atol=1.0e-6)

            # intersect
            p₁ = Point_3D(T, 0)
            p₂ = Point_3D(T, 2)
            p₃ = Point_3D(T, 2, 2)
            p₄ = Point_3D(T, 1, 1//4)
            p₅ = Point_3D(T, 3, 1)
            p₆ = Point_3D(T, 1, 1)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))
            l = LineSegment_3D(Point_3D(T, 1, 1//2, -2),
                            Point_3D(T, 1, 1//2,  2))
            intersection = l ∩ tri6

            # 1 intersection
            @test intersection[1]
            @test intersection[2] == 1
            @test intersection[3] ≈ Point_3D(T, 1, 1//2, 0)

            # 0 intersection
            l = LineSegment_3D(Point_3D(T, 1, 0, -2),
                            Point_3D(T, 1, 0,  2))
            intersection = l ∩ tri6
            @test !intersection[1]
            @test intersection[2] == 0

            # 2 intersections
            p₂ = Point_3D(T, 2, 0, 3)
            tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))

            l = LineSegment_3D(Point_3D(T, 1, -2, -1//5),
                               Point_3D(T, 1,  2, -1//5))
            intersection = l ∩ tri6
            @test intersection[1]
            @test intersection[2] == 2
            @test norm(intersection[3] - Point_3D(T, 1, 0.313186813, -1//5)) < 1e-6
            @test norm(intersection[4] - Point_3D(T, 1, 0.766071428, -1//5)) < 1e-6
        end
    end
end
