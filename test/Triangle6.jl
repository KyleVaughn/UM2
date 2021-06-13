using MOCNeutronTransport
@testset "Triangle6" begin
    for T in [Float32, Float64, BigFloat]
        @testset "Constructors" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            p₄ = Point(T, 1//2)
            p₅ = Point(T, 1, 1//2)
            p₆ = Point(T, 1//2, 1//2)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)

            # single constructor
            tri6 = Triangle6(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)
        end

        @testset "Methods" begin
            p₁ = Point(T, 0)
            p₂ = Point(T, 1)
            p₃ = Point(T, 1, 1)
            p₄ = Point(T, 1//2)
            p₅ = Point(T, 1, 1//2)
            p₆ = Point(T, 1//2, 1//2)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

            # interpolation
            tri6(0, 0) ≈ p₁
            tri6(1, 0) ≈ p₂
            tri6(0, 1) ≈ p₃
            tri6(1//2, 0) ≈ p₄
            tri6(1//2, 1//2) ≈ p₅
            tri6(0, 1//2) ≈ p₆
            tri6(1//2, 1//2) ≈ Point(T, 1//2, 1//2)

            # area
            p₁ = Point(T, 0)
            p₂ = Point(T, 2)
            p₃ = Point(T, 2, 2)
            p₄ = Point(T, 1, 1//4)
            p₅ = Point(T, 3, 1)
            p₆ = Point(T, 1, 1)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

            # 2D default
            @test isapprox(area(tri6; N = 12), 3, atol=1.0e-6)
            # 3D default
            p₂ = Point(T, 2, 0, 3)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            @test isapprox(area(tri6; N = 79), 6.328781460309, atol=1.0e-6)

            # intersect
            p₁ = Point(T, 0)
            p₂ = Point(T, 2)
            p₃ = Point(T, 2, 2)
            p₄ = Point(T, 1, 1//4)
            p₅ = Point(T, 3, 1)
            p₆ = Point(T, 1, 1)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            l = LineSegment(Point(T, 1, 1//2, -2),
                            Point(T, 1, 1//2,  2))
            intersection = l ∩ tri6

            # 1 intersection
            @test intersection[1]
            @test intersection[2] == 1
            @test intersection[3][1] ≈ Point(T, 1, 1//2, 0)

            # 0 intersection
            l = LineSegment(Point(T, 1, 0, -2),
                            Point(T, 1, 0,  2))
            intersection = l ∩ tri6
            @test !intersection[1]
            @test intersection[2] == 0

            # 2 intersections
            p₂ = Point(T, 2, 0, 3)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

            l = LineSegment(Point(T, 1, -2, -1//5),
                            Point(T, 1,  2, -1//5))
            intersection = l ∩ tri6
            @test intersection[1]
            @test intersection[2] == 2
            @test norm(intersection[3][1] - Point(T, 1, 0.766071428, -1//5)) < 1e-6
            @test norm(intersection[3][2] - Point(T, 1, 0.313186813, -1//5)) < 1e-6

            # in
            p₁ = Point(T, 0)
            p₂ = Point(T, 2)
            p₃ = Point(T, 2, 2)
            p₄ = Point(T, 1, 1//4)
            p₅ = Point(T, 3, 1)
            p₆ = Point(T, 1, 1)
            tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))
            @test Point(T, 1, 1//2) ∈ tri6
            @test Point(T, 1, 0) ∉  tri6
        end
    end
end
