using MOCNeutronTransport
@testset "Triangle6_2D" begin
    for F in [Float32, Float64]
        @testset "Constructors" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 1, 1)
            p₄ = Point_2D(F, 1//2)
            p₅ = Point_2D(F, 1, 1//2)
            p₆ = Point_2D(F, 1//2, 1//2)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)

            # single constructor
            tri6 = Triangle6_2D(p₁, p₂, p₃, p₄, p₅, p₆)
            @test tri6.points == (p₁, p₂, p₃, p₄, p₅, p₆)
        end

        @testset "Methods" begin
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 1)
            p₃ = Point_2D(F, 0, 1)
            p₄ = Point_2D(F, 1//2)
            p₅ = Point_2D(F, 1//2, 1//2)
            p₆ = Point_2D(F, 0, 1//2)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))

            # interpolation
            @test tri6(0, 0) ≈ p₁
            @test tri6(1, 0) ≈ p₂
            @test tri6(0, 1) ≈ p₃
            @test tri6(1//2, 0) ≈ p₄
            @test tri6(1//2, 1//2) ≈ p₅
            @test tri6(0, 1//2) ≈ p₆

            # derivative
            r, s = derivative(tri6, 0, 0)
            @test r ≈ Point_2D(F, 1, 0)
            @test s ≈ Point_2D(F, 0, 1)
            r, s = derivative(tri6, 1, 0)
            @test r ≈ Point_2D(F, 1, 0)
            @test s ≈ Point_2D(F, 0, 1)
            r, s = derivative(tri6, 0, 1)
            @test r ≈ Point_2D(F, 1, 0)
            @test s ≈ Point_2D(F, 0, 1)
            r, s = derivative(tri6, 1//2, 1//2)
            @test r ≈ Point_2D(F, 1, 0)
            @test s ≈ Point_2D(F, 0, 1)

            # area
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 2)
            p₄ = Point_2D(F, 3//2, 1//4)
            p₅ = Point_2D(F, 3, 1)
            p₆ = Point_2D(F, 1, 1)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            # 2D default
            @test isapprox(area(tri6; N = 12), 3, atol=1.0e-6)

            # real_to_parametric
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 2)
            p₄ = Point_2D(F, 3//2, 1//4)
            p₅ = Point_2D(F, 3, 1)
            p₆ = Point_2D(F, 1, 1)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test tri6(real_to_parametric(p₁, tri6)) ≈ p₁
            @test tri6(real_to_parametric(p₂, tri6)) ≈ p₂
            @test tri6(real_to_parametric(p₃, tri6)) ≈ p₃
            @test tri6(real_to_parametric(p₄, tri6)) ≈ p₄
            @test tri6(real_to_parametric(p₅, tri6)) ≈ p₅
            @test tri6(real_to_parametric(p₆, tri6)) ≈ p₆

            # in
            p₁ = Point_2D(F, 0)
            p₂ = Point_2D(F, 2)
            p₃ = Point_2D(F, 2, 2)
            p₄ = Point_2D(F, 3//2, 1//4)
            p₅ = Point_2D(F, 3, 1)
            p₆ = Point_2D(F, 1, 1)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            @test Point_2D(F, 1, 1//2) ∈ tri6
            @test Point_2D(F, 1, 0) ∉  tri6

            # intersect
            # 0 intersection
            l = LineSegment_2D(Point_2D(F, 0, -1), Point_2D(F, 4, -1))
            n, points = l ∩ tri6
            @test n == 0

            # 2 intersection
            l = LineSegment_2D(Point_2D(F, 0, 0), Point_2D(F, 4, 0))
            n, points = l ∩ tri6
            @test n == 2
            @test points[1] ≈ Point_2D(F, 0, 0)
            @test points[2] ≈ Point_2D(F, 2, 0)

            l = LineSegment_2D(Point_2D(F, 0, 1), Point_2D(F, 4, 1))
            n, points = l ∩ tri6
            @test n == 2
            @test points[1] ≈ Point_2D(F, 3, 1)
            @test points[2] ≈ Point_2D(F, 1, 1)

            # 4 intersection
            l = LineSegment_2D(Point_2D(F, 0, 1//10), Point_2D(F, 4, 1//10))
            n, points = l ∩ tri6
            @test n == 4
            @test points[1] ≈ Point_2D(F, 0.4254033307585166, 1//10)
            @test points[2] ≈ Point_2D(F, 1.9745966692414834, 1//10)
            @test points[3] ≈ Point_2D(F, 2.1900000000000000, 1//10)
            @test points[4] ≈ Point_2D(F, 1//10,              1//10)

            # 6 intersection
            p₁ = Point_2D(F,  1, 0)
            p₂ = Point_2D(F,  0, 0)
            p₃ = Point_2D(F, -1, 0)
            p₄ = Point_2D(F,  1//2, -1//2)
            p₅ = Point_2D(F, -1//2, -1//2)
            p₆ = Point_2D(F,   0,    -2)
            tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
            l = LineSegment_2D(Point_2D(F, -2, -1//4), Point_2D(F, 2, -1//4))
            n, points = l ∩ tri6
            @test n == 6
            @test points[1] ≈ Point_2D{F}(F[ 0.14644659, -1//4])
            @test points[2] ≈ Point_2D{F}(F[ 0.8535534,  -1//4])
            @test points[3] ≈ Point_2D{F}(F[-0.8535534,  -1//4])
            @test points[4] ≈ Point_2D{F}(F[-0.14644665, -1//4])
            @test points[5] ≈ Point_2D{F}(F[ 0.9354143,  -1//4])
            @test points[6] ≈ Point_2D{F}(F[-0.9354143,  -1//4])
        end
    end
end
